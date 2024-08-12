import os
import cv2
import time
import tqdm
import glob
import time
import rembg
import torch
import imageio
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
from cam_utils import orbit_camera, OrbitCamera
from gs_renderer import Renderer, MiniCam
from torchvision.utils import save_image
from PIL import Image
from pathlib import Path
from knn_cuda import KNN
import pytorch3d.ops as ops
from chamferdist import ChamferDistance

class GUI:
    def __init__(self, opt):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)
        
        # self.seed = 0
        # self.seed_everything()

        # models
        self.device = torch.device("cuda")
        self.bg_remover = None

        self.guidance_zero123 = None
        self.enable_zero123 = False
        
        if self.opt.train_dynamic:
            image_list = glob.glob(self.opt.input_folder + '/' + '*.png')
            image_list = sorted(image_list, key=lambda x: int(x.split('.')[0].split('/')[-1]))
            self.num_frames = len(image_list) # default: 32
            print("{} frames to load!!!".format(self.num_frames))

        # renderer
        self.renderer = Renderer(sh_degree=self.opt.sh_degree)
        self.test_renderer = Renderer(sh_degree=self.opt.sh_degree)

        # gt
        self.source_images = []
        self.source_masks = []
        self.source_time = []

        # training stuff
        self.optimizer = None
        self.step = 0
        self.train_steps = 1  # steps per rendering loop
        self.stage = "s1"
        
        if self.opt.train_dynamic:
            for i in range(len(image_list)):
                mask, image = self.load_input(image_list[i])
                self.source_images.append(image)
                self.source_masks.append(mask)
                self.source_time.append(i/len(image_list))
                
        if self.opt.train_dynamic:
            # override if provide a checkpoint
            if self.opt.load_stage == "s1":
                save_path = os.path.join(self.opt.save_path, "s1/point_cloud.ply")
                g = self.renderer.gaussians
                g.load_ply(save_path)
                self.renderer.initialize(num_pts=g._xyz.shape[0], num_cpts=g._xyz.shape[0])           
            else:
                # initialize gaussians to a blob
                self.renderer.initialize(num_pts=self.opt.num_cpts, num_cpts=self.opt.num_cpts)
        
        self.chamferDist = ChamferDistance()
        self.cpts_s1 = []

    def seed_everything(self):
        try:
            seed = int(self.seed)
        except:
            seed = np.random.randint(0, 1000000)

        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        self.last_seed = seed

    def prepare_train(self):

        self.step = 0
        self.stage = "s1"
        self.opt.position_lr_max_steps = 500

        # setup training
        self.renderer.gaussians.training_setup(self.opt)
        # do not do progressive sh-level
        self.renderer.gaussians.active_sh_degree = self.renderer.gaussians.max_sh_degree
        self.optimizer = self.renderer.gaussians.optimizer

        if self.stage == "s1":
            for param_group in self.optimizer.param_groups:
                if param_group["name"] == "c_radius":
                    param_group['lr'] = 0.0
                if param_group["name"] == "c_xyz":
                    param_group['lr'] = 0.0

        # default camera
        pose = orbit_camera(self.opt.elevation, 0, self.opt.radius)
        self.fixed_cam = MiniCam(
            pose,
            self.opt.ref_size,
            self.opt.ref_size,
            self.cam.fovy,
            self.cam.fovx,
            self.cam.near,
            self.cam.far,
        )

        self.enable_zero123 = self.opt.lambda_zero123 > 0 and len(self.source_images) > 0

        if self.guidance_zero123 is None and self.enable_zero123:
            print(f"[INFO] loading zero123...")
            from guidance.zero123_utils import Zero123
            self.guidance_zero123 = Zero123(self.device)
            print(f"[INFO] loaded zero123!")

    def train_step(self):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()
        
        if self.stage == "s1" and self.step == self.opt.FPS_iter:
            self.FPS(num_pts=self.opt.num_cpts)

        if self.stage == "s2" and self.step == 0:
            c_means3D = self.renderer.gaussians._c_xyz
            for t in self.source_time:
                means3D_deform, _ = self.renderer.gaussians._timenet(c_means3D, t) 
                self.cpts_s1.append(c_means3D + means3D_deform)
        
        for _ in range(self.train_steps):

            self.step += 1
            
            if self.stage == "s1":
                iters = self.opt.iters_s1
            elif self.stage == "s2":
                iters = self.opt.iters_s2
            else:
                assert ValueError("Video-to-4D generation of SC4D only contain two stages!!!")
            step_ratio = self.step / iters
            
            self.renderer.gaussians.update_learning_rate(self.step, self.stage)
            
            if self.stage == "s2" and self.step < 1000:
                for param_group in self.optimizer.param_groups:
                    if param_group["name"] == "xyz":
                        param_group['lr'] = 0.0002 
            
            # find knn
            if self.stage >= "s2":
                self.find_knn(g=self.renderer.gaussians, k=4)   
        
            loss = 0
        
            # random reference index
            index = np.random.randint(0, len(self.source_images))
                        
            self.input_img_torch = self.source_images[index]
            self.input_mask_torch = self.source_masks[index]
            self.timestamp = self.source_time[index]
            if self.stage == "s2":
                self.cpts_ori = self.cpts_s1[index]

            with torch.no_grad():
                if self.enable_zero123:
                    self.guidance_zero123.get_img_embeds(self.input_img_torch)

            render_resolution = 128 if self.step < 200 else (256 if self.step < 300 else 512)
            pose = orbit_camera(self.opt.elevation, 0, self.opt.radius)
            self.fixed_cam = MiniCam(
                pose,
                render_resolution,
                render_resolution,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
            )

            ### known view
            if self.input_img_torch is not None:
                cur_cam = self.fixed_cam
                out = self.renderer.render(cur_cam, time=self.timestamp, stage=self.stage)
                
                if self.opt.add_ga and self.stage == "s2":
                    cpts_ori = self.cpts_ori.detach()
                    cpts = out["cpts_t"]
                    if self.opt.ga_chamfer:
                        dist_forward = self.chamferDist(cpts[None, ...], cpts_ori[None, ...]) # the order is important!!!
                        loss = loss + self.opt.lambda_ga1 * dist_forward
                    else:
                        loss = loss + self.opt.lambda_ga2 * (cpts - cpts_ori).abs().mean()
                
                # image loss
                image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                input_img_torch = F.interpolate(self.input_img_torch, (render_resolution, render_resolution), mode="bilinear", align_corners=False)
                loss = loss + 5000 * F.mse_loss(image, input_img_torch)

                # mask loss
                mask = out["alpha"].unsqueeze(0) # [1, 1, H, W] in [0, 1]
                input_mask_torch = F.interpolate(self.input_mask_torch, (render_resolution, render_resolution), mode="bilinear", align_corners=False)
                loss = loss + 500 * F.mse_loss(mask, input_mask_torch)

            ### novel view (manual batch)
            images = []
            depths = []
            poses = []
            vers, hors, radii = [], [], []
            # avoid too large elevation, and make sure it always cover [-min_ver, min_ver]
            min_ver = max(min(self.opt.min_ver, self.opt.min_ver - self.opt.elevation), -80 - self.opt.elevation)
            max_ver = min(max(self.opt.max_ver, self.opt.max_ver - self.opt.elevation), 80 - self.opt.elevation)                                 
            
            for _ in range(self.opt.batch_size):
                # render random view
                ver = np.random.randint(min_ver, max_ver)
                hor = np.random.randint(-180, 180)
                radius = 0

                vers.append(ver)
                hors.append(hor)
                radii.append(radius)

                pose = orbit_camera(self.opt.elevation + ver, hor, self.opt.radius + radius)
                poses.append(pose)

                cur_cam = MiniCam(pose, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)

                bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.opt.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device="cuda")
                out = self.renderer.render(cur_cam, bg_color=bg_color, time=self.timestamp, stage=self.stage)

                image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                depth = out["depth"].unsqueeze(0)

                images.append(image)
                depths.append(depth)
            
            images = torch.cat(images, dim=0)
            depths = torch.cat(depths, dim=0)
            poses = torch.from_numpy(np.stack(poses, axis=0)).to(self.device)
            
            if self.stage == "s1":
                min_step_t = self.opt.t_range_s1[0]
                max_step_t = self.opt.t_range_s1[1]
                self.guidance_zero123.set_min_max_steps(min_step_t, max_step_t)
            elif self.stage == "s2":
                min_step_t = self.opt.t_range_s2[0]
                max_step_t = self.opt.t_range_s2[1]
                self.guidance_zero123.set_min_max_steps(min_step_t, max_step_t)
            
            if self.enable_zero123:
                loss = loss + self.opt.lambda_zero123 * self.guidance_zero123.train_step(images, vers, hors, radii, step_ratio=step_ratio, default_elevation=self.opt.elevation)

            if self.opt.use_arap and self.stage == "s1" and self.step > self.opt.arap_start_iter:
                loss_arap, conns = self.renderer.arap_loss_v2(stage=self.stage)
                loss += self.opt.lambda_arap * loss_arap
            
            if self.opt.use_arap and self.stage == "s2" and self.step < 2000:
                loss_arap, conns = self.renderer.arap_loss_v2(stage=self.stage)
                loss += self.opt.lambda_arap * loss_arap
                   
            with torch.no_grad():
                if self.opt.do_inference and (self.step - 1) % self.opt.check_inter == 0:
                    self.test_3d()
                
            # optimize step
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.opt.save_inter == 0:
                save_path = os.path.join(self.opt.save_path, self.stage)
                path2 = os.path.join(save_path, "point_cloud_c_{}.ply".format(self.step)) if self.stage >= "s2" else None
                self.renderer.gaussians.save_ply(os.path.join(save_path, "point_cloud_{}.ply".format(self.step)), path2)
                self.renderer.gaussians.save_model(save_path, step=self.step)

            # densify and prune
            if self.stage == "s1":
                if self.step >= self.opt.density_start_iter and self.step <= self.opt.density_end_iter:
                    viewspace_point_tensor, visibility_filter, radii = out["viewspace_points"], out["visibility_filter"], out["radii"]
                    self.renderer.gaussians.max_radii2D[visibility_filter] = torch.max(self.renderer.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    self.renderer.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if self.step % self.opt.densification_interval == 0:
                        self.renderer.gaussians.densify_and_prune(self.opt.densify_grad_threshold, min_opacity=0.01, extent=4, max_screen_size=1)
                        print("Num of gaussians: ", self.renderer.gaussians._xyz.shape[0])

                    if self.step % self.opt.opacity_reset_interval == 0:
                        self.renderer.gaussians.reset_opacity()
            
            if self.stage == "s2":
                if self.step % self.opt.densification_interval_s2 == 0 and self.opt.init_type == "ag":
                    self.renderer.gaussians.prune(min_opacity=0.01, extent=4, max_screen_size=1)
                    print("Num of gaussians after pruning: ", self.renderer.gaussians._xyz.shape[0])

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)
    
    def load_input(self, file):
        # load image
        print(f'[INFO] load image from {file}...')
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if img.shape[-1] == 3:
            if self.bg_remover is None:
                self.bg_remover = rembg.new_session()
            img = rembg.remove(img, session=self.bg_remover)

        img = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0

        input_mask = img[..., 3:]
        # white bg
        input_img = img[..., :3] * input_mask + (1 - input_mask)
        # bgr to rgb
        input_img = input_img[..., ::-1].copy()
        
        # to torch tensors
        input_img_torch = torch.from_numpy(input_img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        input_img = F.interpolate(input_img_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)
        input_mask_torch = torch.from_numpy(input_mask).permute(2, 0, 1).unsqueeze(0).to(self.device)
        input_mask = F.interpolate(input_mask_torch, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)

        return input_mask, input_img

    def find_knn(self, g, k=4):
        control_pts = g._c_xyz.detach()
        gaussian_pts = g._xyz.detach()
        knn = KNN(k=k, transpose_mode=True)
        dist, indx = knn(control_pts.unsqueeze(0), gaussian_pts.unsqueeze(0))  # 32 x 50 x 10
        dist, indx = dist[0], indx[0]  
        g.neighbor_dists = dist             
        g.neighbor_indices = indx

    def FPS(self, num_pts):
        g = self.renderer.gaussians
        _, idxs = ops.sample_farthest_points(points=g._xyz.unsqueeze(0), K=num_pts)
        idxs = idxs[0]
        g.prune_points(idxs)
    
    def load_model(self, g):
        load_stage = self.opt.load_stage or self.opt.test_stage
        path1 = "{}/{}/point_cloud.ply".format(self.opt.save_path, load_stage)
        path2 = "{}/{}/point_cloud_c.ply".format(self.opt.save_path, load_stage)
        model_dir = "{}/{}".format(self.opt.save_path, load_stage)
        if self.opt.test_step:
            path1 = path1.split('.')[0] + "_{}".format(self.opt.test_step) + '.ply'
            if test_stage > "s1":
                path2 = path2.split('.')[0] + "_{}".format(self.opt.test_step) + '.ply'
        if load_stage < "s2":
            path2 = None
        g.load_ply(path1, path2)
        g.load_model(model_dir, self.opt.test_step)

    def test_3d(self, test_cpts=True, render_type="fixed"):
        video_save_dir = self.opt.video_save_dir
        if not os.path.exists(video_save_dir):
            os.makedirs(video_save_dir)
        frames = []
        init_ver = 0
        if test_cpts:
            self.test_cpts(test_stage=self.stage, render_type=render_type)
        for i in range(32):
            pose = orbit_camera(0, init_ver, self.opt.radius)
            cur_cam = MiniCam(
                    pose,
                    self.W,
                    self.H,
                    self.cam.fovy,
                    self.cam.fovx,
                    self.cam.near,
                    self.cam.far,
                )
            out = self.renderer.render(cur_cam, time=i/32, stage=self.stage)
            img = out["image"].detach().cpu().permute(1, 2, 0).numpy() * 255
            img = img.astype('uint8')
            frames.append(img)
        # compose video
        save_name = self.opt.save_path.split("/")[-1].split(".")[0]
        video_name = video_save_dir + '/{}.mp4'.format(save_name)
        imageio.mimwrite(video_name, frames, fps=10, quality=8, macro_block_size=1)
    
    def test(self, test_cpts=True, render_type="fixed"):
        video_save_dir = self.opt.video_save_dir
        test_stage = self.opt.test_stage
        if not os.path.exists(video_save_dir):
            os.makedirs(video_save_dir)
        frames = []
        g = self.renderer.gaussians
        self.load_model(g=g)
        if test_stage >= "s2":
            self.find_knn(g)
        if test_cpts:
            self.test_cpts(test_stage=self.opt.test_stage, render_type=render_type)
        for i in range(32):
            if render_type == "fixed":
                test_azi = self.opt.test_azi
            else:
                test_azi = 360/32*i
            pose = orbit_camera(0, test_azi, self.opt.radius)
            cur_cam = MiniCam(
                    pose,
                    self.W,
                    self.H,
                    self.cam.fovy,
                    self.cam.fovx,
                    self.cam.near,
                    self.cam.far,
                )
            out = self.renderer.render(cur_cam, time=i/32, stage=test_stage)
            img = out["image"].detach().cpu().permute(1, 2, 0).numpy() * 255
            img = img.astype('uint8')
            frames.append(img)
        save_name = self.opt.save_path.split("/")[-1].split(".")[0]
        if render_type == "fixed":
            video_name = video_save_dir + '/{}_{}.mp4'.format(save_name, self.opt.test_azi)
        else:
            video_name = video_save_dir + '/{}_circle.mp4'.format(save_name)
        imageio.mimwrite(video_name, frames, fps=10, quality=8, macro_block_size=1)

    def test_cpts(self, test_stage="s1", render_type="fixed", sh_degree=0):
        video_save_dir = self.opt.video_save_dir
        renderer = Renderer(sh_degree=sh_degree)
        if test_stage > "s1":
            renderer.initialize(num_pts=self.renderer.gaussians._c_xyz.shape[0])
            renderer.gaussians._xyz = self.renderer.gaussians._c_xyz
        else:
            renderer.initialize(num_pts=self.renderer.gaussians._xyz.shape[0])
            renderer.gaussians._xyz = self.renderer.gaussians._xyz
        renderer.gaussians._r = torch.ones((1), device="cuda", requires_grad=True) * -5.0
        renderer.gaussians._timenet = self.renderer.gaussians._timenet
        num_pts = renderer.gaussians._xyz.shape[0]
        device = renderer.gaussians._xyz.device
        renderer.gaussians._scaling = torch.ones((num_pts, 3), device=device, requires_grad=True) * -5.0
        renderer.gaussians._opacity = torch.ones((num_pts, 1), device=device, requires_grad=True) * 2.0
        color = torch.ones((num_pts, 3), device=device) * 0.1
        frames = []
        init_ver = 0
        ###
        cpts_tra = 0
        for i in range(32):
            if render_type == "fixed":
                test_azi = self.opt.test_azi
            else:
                test_azi = 360/32*i
            pose = orbit_camera(0, test_azi, self.opt.radius)
            cur_cam = MiniCam(
                    pose,
                    self.W,
                    self.H,
                    self.cam.fovy,
                    self.cam.fovx,
                    self.cam.near,
                    self.cam.far,
                )
            out = renderer.render(cur_cam, override_color=color, time=i/32, stage="s1")
            img = out["image"].detach().cpu().permute(1, 2, 0).numpy() * 255
            img = img.astype('uint8')
            frames.append(img)
            ###
            if i == 0:
                cpts_tmp = out["cpts_t"]
            cpts_t = out["cpts_t"]
            cpts_tra += torch.dist(cpts_t, cpts_tmp, p=2)
            cpts_tmp = cpts_t
        print("cpts average moving length: ", cpts_tra.item()) 
        ###
        save_name = self.opt.save_path.split("/")[-1].split(".")[0]
        if render_type == "fixed":
            video_name = video_save_dir + '/{}_cpts_{}.mp4'.format(save_name, self.opt.test_azi)
        else:
            video_name = video_save_dir + '/{}_cpts_circle.mp4'.format(save_name)
        imageio.mimwrite(video_name, frames, fps=10, quality=8, macro_block_size=1)

    def prepare_train_s2(self):
        self.stage = "s2"
        self.step = 0
        g = self.renderer.gaussians
        if self.opt.load_stage == "":
            with torch.no_grad():
                g._c_xyz.copy_(g._xyz) 
                g._scaling.copy_(g._r.expand_as(g._xyz))
                g._c_radius.copy_(g._r.expand_as(g._c_radius))
            if self.opt.init_type == "normal":
                self.renderer.initialize(num_pts=self.opt.num_pts, only_init_gaussians=True)
            elif self.opt.init_type == "ag":
                self.renderer.initialize_ag(g._c_xyz, g.get_c_radius(stage="s2"), num_cpts=g._c_xyz.shape[0], num_pts_per_cpt=200, init_ratio=self.opt.init_ratio)
            else:
                raise ValueError("Unsupported init type!!!")
            # update training
            self.renderer.gaussians.training_setup(self.opt)
            self.renderer.gaussians.active_sh_degree = self.renderer.gaussians.max_sh_degree
            self.optimizer = self.renderer.gaussians.optimizer
        g._r = torch.tensor([], device="cuda")
        r_id = 0
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "r":
                param_group['lr'] = 0.0
                self.optimizer.param_groups.pop(r_id)
            r_id += 1
        self.opt.position_lr_max_steps = self.opt.iters_s2
        self.opt.position_lr_init = 0.0002
        self.opt.position_lr_final = 0.000002

    def train_dynamic(self, iters_s1=1500, iters_s2=5000, load_stage=""): 
        g = self.renderer.gaussians
        iters_s1 = iters_s1 if load_stage < "s1" else 0
        iters_s2 = iters_s2 if load_stage < "s2" else 0
        if load_stage != "": 
            print("Loading from stage {}...".format(load_stage[-1]))
            # Loading pretrained model
            self.load_model(g=g)
            if load_stage >= "s1":
                g._r.data = g._scaling
                if load_stage == "s1":
                    with torch.no_grad():
                        g._c_xyz.copy_(g._xyz) 
                        g._c_radius.copy_(g._scaling.mean(dim=1, keepdim=True))
                    if self.opt.init_type == "normal":
                        self.renderer.initialize(num_pts=self.opt.num_pts, only_init_gaussians=True)
                    elif self.opt.init_type == "ag":
                        self.renderer.initialize_ag(g._c_xyz, g.get_c_radius(stage="s2"), num_cpts=g._c_xyz.shape[0], num_pts_per_cpt=200, init_ratio=self.opt.init_ratio)
                    else:
                        raise ValueError("Unsupported init type!!!")
        
        self.opt.save_path = self.opt.save_path if self.opt.save_path_new is None else self.opt.save_path_new

        ### Stage 1: coarse stage of video-to-4D generation 
        self.prepare_train()
        self.renderer.gaussians.lr_setup(self.opt)
        if iters_s1 > 0:
            for i in tqdm.trange(iters_s1):
                self.train_step()
            # prune points at s1 end
            self.renderer.gaussians.prune_s1_end(min_opacity=0.01, extent=4, max_screen_size=1)
            print("Num of cpts after s1: ", self.renderer.gaussians._c_xyz.shape[0])
            # save s1
            save_path = os.path.join(self.opt.save_path, "s1")
            g.save_ply(os.path.join(save_path, "point_cloud.ply"))
            g.save_model(save_path)

        ### Stage 2: fine stage of video-to-4D generation 
        self.prepare_train_s2()
        self.renderer.gaussians.lr_setup(self.opt)
        if iters_s2 > 0:
            for i in tqdm.trange(iters_s2):
                self.train_step()
            # save s2
            save_path = os.path.join(self.opt.save_path, "s2")
            g.save_ply(os.path.join(save_path, "point_cloud.ply"), os.path.join(save_path, "point_cloud_c.ply"))
            g.save_model(save_path)

if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/sc4d.yaml", required=False, help="path to the yaml config file")
    args, extras = parser.parse_known_args()

    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))

    gui = GUI(opt)

    if opt.train_dynamic:
        gui.train_dynamic(opt.iters_s1, opt.iters_s2, opt.load_stage)
    else:
        gui.test(render_type=opt.render_type)
