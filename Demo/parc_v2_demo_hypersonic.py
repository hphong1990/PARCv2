import numpy as np
import tensorflow as tf
import parc.model.model_hypersonic as model
import argparse

# Create tf.dataset
def load_tfdataset(mach_list, seq=2, len_load=120):
    for idx, each_mach in enumerate(mach_list):
        tmp = np.load("data/normalized/mach_%.1f.npy" % each_mach)
        if len_load == 40:
            tmp = tmp[::3, :, :, :]
        tfds_tmp = tf.data.Dataset.from_tensor_slices(tmp)
        tfds_tmp = tfds_tmp.window(seq, shift=1, drop_remainder=True)
        tfds_tmp = tfds_tmp.flat_map(lambda window: window.batch(seq))
        tfds_tmp = tfds_tmp.map(lambda window: 
                                ((window[0, :, :, :2], window[0, :, :, 2:]),
                                 (window[1:, :, :, :2], window[1:, :, :, 2:])))
        if idx == 0:
            tfds = tfds_tmp
        else:
            tfds = tfds.concatenate(tfds_tmp)
    return tfds


# Save best test performance
class PARCBestValLoss(tf.keras.callbacks.Callback):
    def __init__(self, save_diff, save_int):
        super().__init__()
        self.save_diff = save_diff
        self.save_int = save_int
        self.best = np.inf

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get("val_val_loss")
        if self.best > val_loss:
            self.best = val_loss
            if self.save_diff:
                self.model.differentiator.save_weights(args.save_diff)
            if self.save_int:
                self.model.integrator.save_weights(args.save_int)


if __name__ == "__main__":
    # Args parsing
    parser = argparse.ArgumentParser(prog='ProgramName', description='Training script for PARCv2 on hypersonic data')
    parser.add_argument('--batch', type=int, default=1, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-5, help="Learn rate")
    parser.add_argument('--nsteps', type=int, default=120, help="Timesteps of the training simulation data")
    parser.add_argument('--epoch', type=int, default=100, help="Epoch to train for")
    parser.add_argument('--load_diff', type=str, default="", help="File to load weights of differentiator before training from. Keep empty to start from scratch.")
    parser.add_argument('--load_int', type=str, default="", help="File to load weights of integrator before training from. Keep empty to start from scratch.")
    parser.add_argument('--train', nargs='+', help='<Required> Mach number to use as training set', required=True, type=float)
    parser.add_argument('--val', nargs='+', help='<Required> Mach number to use as validation set', required=True, type=float)
    parser.add_argument('--mode', type=str, default="differentiator_training", help="Mode selection: differentiator_training, integrator_training")
    parser.add_argument('--seq_len', type=int, default=2, help="Sequence length for training")
    parser.add_argument('--solver', type=str, default="rk4", help="Solver to use during training")
    parser.add_argument('--save_diff', type=str, default="", help="File to save weights of differentiator to. Keep empty to not save.")
    parser.add_argument('--save_int', type=str, default="", help="File to save weights of integrator to. Keep empty to not save.")
    args = parser.parse_args()
    # Dataset
    diff_train = load_tfdataset(args.train, args.seq_len, args.nsteps).shuffle(buffer_size=2192).batch(args.batch)
    diff_test = load_tfdataset(args.val, args.seq_len, args.nsteps).shuffle(buffer_size=2192).batch(args.batch)
    # Training
    pbvl = PARCBestValLoss(args.save_diff, args.save_int)
    step_size = 1.0 / args.nsteps 
    tf.keras.backend.clear_session()
    parc = model.PARCv2(n_state_var=2, n_time_step=args.seq_len-1, step_size=step_size, solver=args.solver, mode=args.mode)
    parc.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, beta_1=0.9, beta_2=0.999))
    if args.load_diff:
        parc.differentiator.load_weights(args.load_diff)
    if args.load_int:
        parc.integrator.load_weights(args.load_int)
    parc.fit(diff_train, epochs=args.epoch, shuffle=True, validation_data=diff_test, callbacks=[pbvl])
