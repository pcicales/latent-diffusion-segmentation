def run_tensorboard(logdir_absolute):

   import os, threading
   tb_thread = threading.Thread(
          target=lambda: os.system('/home/cougarnet.uh.edu/pcicales/anaconda3/envs/ldm/bin/tensorboard '
                                   '--logdir=' + logdir_absolute),
          daemon=True)
   tb_thread.start()

if __name__ == "__main__":
    run_tensorboard('/data/pcicales/latent_diffusion_segmentation/autoencoder_camelyon_superres/2022-10-01T17-21-45_autoencoder_kl_55x110x3_camelyon/testtube/version_0/tf')