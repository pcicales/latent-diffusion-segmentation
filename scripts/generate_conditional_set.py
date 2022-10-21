import argparse, os, sys, glob, datetime, yaml
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config

from ldm.models.diffusion.ddim import DDIMSampler

import numpy as np
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid


#####################
###### HELPERS ######
#####################

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location=torch.device('cpu'))
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model

def get_config(conf_dir):
    print(f"Loading config from {conf_dir}")
    config = OmegaConf.load(conf_dir)
    return config

def get_model(config, mod):
    model = load_model_from_config(config, mod)
    return model

def get_parser():
    def str2list(v):
        if isinstance(v, list):
            return v
        elif isinstance(v, str):
            if (v[0] == '[') and (v[-1] == ']'):
                v = v.strip('][').split(', ')
            else:
                v = v.split(', ')
            try:
                v = [int(i) for i in v]
            except:
                raise argparse.ArgumentTypeError("Class labels must be integers")
            return v

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--n_sample_sets",
        type=int,
        nargs="?",
        help="Number of sample sets to draw per class.",
        default=2
    )
    parser.add_argument(
        "-ni",
        "--n_sample_set_images",
        type=int,
        nargs="?",
        help="Number of sample set instances to draw per class.",
        default=64
    )
    parser.add_argument(
        "-c",
        "--classes",
        type=str2list,
        nargs="?",
        help="Classes that we wish to sample from.",
        default='[0, 1]'
    )
    parser.add_argument(
        "-dds",
        "--ddim_steps",
        type=int,
        nargs="?",
        help="Number of sampling steps, increasing ddim_steps generally "
             "yields higher quality samples, but returns are diminishing for values > 250.",
        default=250
    )
    parser.add_argument(
        "-dde",
        "--ddim_eta",
        type=float,
        nargs="?",
        help="Fast sampling (i e. low values of ddim_steps) while "
             "retaining good quality can be achieved by using ddim_eta = 0.",
        default=0.0
    )
    parser.add_argument(
        "-s",
        "--scale",
        type=float,
        nargs="?",
        help="As a rule of thumb, higher values of scale produce better "
             "samples at the cost of a reduced output diversity.",
        default=3.0
    )
    parser.add_argument(
        "-gm",
        "--grid_mode",
        type=bool,
        nargs="?",
        help="Generate grids of each set and save them for visualization.",
        default=True
    )
    parser.add_argument(
        "-ss",
        "--save_set",
        type=bool,
        nargs="?",
        help="Whether or not to save sets that are generated.",
        default=True
    )
    parser.add_argument(
        "-ssd",
        "--save_set_dir",
        type=str,
        nargs="?",
        help="Save sets in corresponding directory, sub directory for each set.",
        default='/data/public/public_access/CAMELYON16/LDM_sets'
    )
    parser.add_argument(
        "-con",
        "--config",
        type=str,
        nargs="?",
        help="Config .yaml file location for our model.",
        default='/data/pcicales/latent_diffusion_segmentation/set_ldm_camelyon_kl_32x32x3/2022-10-10T18-23-51_camelyon-set-ldm-kl-32x32x3/configs/2022-10-10T18-23-51-project.yaml'
    )
    parser.add_argument(
        "-mod",
        "--model",
        type=str,
        nargs="?",
        help="Model checkpoint .ckpt location.",
        default='/data/pcicales/latent_diffusion_segmentation/set_ldm_camelyon_kl_32x32x3/2022-10-10T18-23-51_camelyon-set-ldm-kl-32x32x3/checkpoints/epoch=000289.ckpt'
    )
    parser.add_argument(
        "-uc",
        "--uc_in",
        type=bool,
        nargs="?",
        help="Whether or not to use unconditional encoding.",
        default=True
    )
    parser.add_argument(
        "-res",
        "--resolution",
        type=str2list,
        nargs="?",
        help="The desired output resolution and channels for each image (h, w).",
        default='[256, 256]'
    )
    parser.add_argument(
        "-sstr",
        "--savestr",
        type=str,
        nargs="?",
        help="Additional savestring for saving sets.",
        default='camelyon256_to_256'
    )

    return parser

#####################
###### MAIN #########
#####################
if __name__ == "__main__":
    #####################
    ##### SETUP #########
    #####################
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    # sys.path.append(os.getcwd())

    parser = get_parser() # get the arg parser
    opt, unknown = parser.parse_known_args() # parse the args

    config = get_config(opt.config) # load the config

    out_res = opt.resolution # the desired output resolution
    # get the encoder output size, which is fed to LDM (input)
    in_res = [config.model.params.channels,
              out_res[0] // (2 ** np.sum(config.model.params.first_stage_config.params.ddconfig.down_mult)),
              out_res[1] // (2 ** np.sum(config.model.params.first_stage_config.params.ddconfig.down_mult))]

    # make config edits as needed
    # image size
    config.model.params.image_size = in_res[0] # currently only supports square, needs to change
    # unet input size
    config.model.params.unet_config.params.image_size = in_res[0] # currently only supports square, needs to change
    # unet input size
    config.model.params.first_stage_config.params.ddconfig.resolution = out_res[0] # currently only supports square, needs to change

    model = get_model(config, opt.model) # get the trained model

    sampler = DDIMSampler(model) # get the sampler

    classes = opt.classes  # define classes to be sampled here

    n_sample_sets = opt.n_sample_sets # define the number of sets we wish to generate
    n_sample_set_images = opt.n_sample_set_images # define the number of instances per sample

    ddim_steps = opt.ddim_steps # define the number of steps
    ddim_eta = opt.ddim_eta # define the eta, results in faster samples at higher quality when 0
    scale = opt.scale # for unconditional guidance, higher values reduce diversity but improve quality

    #####################
    ###### DIRS #########
    #####################
    # make dirs as needed
    if opt.grid_mode:
        # main dir
        main_dir = os.path.join(opt.save_set_dir, now + '___' + opt.config.split('.yaml')[0].split('/')[-1] + '___' + opt.savestr)
        os.makedirs(main_dir, exist_ok=True)

        # grid dir
        grid_dir = os.path.join(main_dir, 'grids')
        os.makedirs(grid_dir, exist_ok=True)

        # class dirs
        class_grid_dir = os.path.join(grid_dir, 'class_{}')
        for class_label in classes:
            os.makedirs(class_grid_dir.format(class_label), exist_ok=True)

    if opt.save_set:
        # main dir
        main_dir = os.path.join(opt.save_set_dir, now + '___' + opt.config.split('.yaml')[0].split('/')[-1] + '___' + opt.savestr)
        os.makedirs(main_dir, exist_ok=True)

        # sets dir
        sets_dir = os.path.join(main_dir, 'sets')
        os.makedirs(sets_dir, exist_ok=True)

        # main class dirs, set dirs
        class_set_dir = os.path.join(sets_dir, 'class_{}')
        set_sample_dir = os.path.join(class_set_dir, 'sample_{}')

        # class dirs
        for class_label in classes:
            os.makedirs(class_set_dir.format(class_label), exist_ok=True)

            # samples dirs
            for sample_id in range(n_sample_sets):
                os.makedirs(set_sample_dir.format(class_label, sample_id + 1), exist_ok=True)

    #####################
    ##### RENDER ########
    #####################
    with torch.no_grad():
        with model.ema_scope():
            if opt.uc_in:
                # fix this, not for ohe, seems they embed imagenet classes differently
                uc = model.get_learned_conditioning(
                    {model.cond_stage_key: torch.tensor(n_sample_set_images * [model.cond_stage_model.n_classes - 1]).to(model.device)}
                )
            else:
                uc = None

            for class_label in classes:
                for sample_id in range(n_sample_sets):
                    print(
                        f"Rendering sample {sample_id+1} (of {n_sample_sets}) with {n_sample_set_images} "
                        f"instances of class '{class_label}' in {ddim_steps} steps (scaling factor of {scale:.2f})."
                    )

                    xc = torch.tensor(n_sample_set_images * [class_label])
                    c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})

                    samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                     conditioning=c,
                                                     batch_size=n_sample_set_images,
                                                     shape=in_res,
                                                     verbose=False,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=uc,
                                                     eta=ddim_eta)

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0,
                                                 min=0.0, max=1.0)

                    if opt.grid_mode:
                        print('Saving grids for class {}, sample {}'.format(class_label, sample_id + 1))
                        grid = make_grid(x_samples_ddim, nrow=int(n_sample_set_images**0.5))
                        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                        grid_pil = Image.fromarray(grid.astype(np.uint8))
                        grid_pil.save(class_grid_dir.format(class_label) + '/sample_{}.png'.format(sample_id + 1))

                    if opt.save_set:
                        print('Saving set instances for class {}, sample {}'.format(class_label, sample_id + 1))
                        for iid, instance in enumerate(x_samples_ddim):
                            ins = 255. * rearrange(instance, 'c h w -> h w c').cpu().numpy()
                            ins = Image.fromarray(ins.astype(np.uint8))
                            ins.save(set_sample_dir.format(class_label, sample_id + 1) + '/seed_{}.png'.format(iid + 1))
