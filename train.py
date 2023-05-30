import argparse
import models
import data
import torch
import time
from pathlib import Path
import tqdm
import json
import utils
import numpy as np
import random
import os
import copy
import configargparse
import shutil

import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

import model_utls
from ddsp import losses

tqdm.monitor_interval = 0

def train(args, network, device, train_sampler, optimizer, ss_weights_dict, epoch, writer):
    loss_container = utils.AverageMeter()
    network.train()
    if args.loss_lsf_weight > 0: network.return_lsf = True
    if args.ss_loss_weight > 0: network.return_synth_controls = True
    if args.supervised: network.return_sources = True
    pbar = tqdm.tqdm(train_sampler, disable=args.quiet)

    if args.loss_voices_weight > 0:
        masks_batch = utils.get_training_masks(batch_mask_shape=(args.batch_size, args.n_sources, 360, 344), 
                                        freq_bands=[(179, 302), (152, 274), (119, 241), (77, 180)], 
                                        device=device)
    
        # plot masks - test debug
        # for i in range(4):
        #     plt.imshow(masks_batch[0, i].detach().cpu().numpy(), origin='lower', aspect='auto', cmap='magma')
        #     plt.colorbar()
        #     plt.savefig('./test_fig/mask_{}_torch.pdf'.format(i))
        #     plt.close()
    
    for data in pbar:
        pbar.set_description("Training batch")
        x = data[0]  # mix
        f0 = data[1]  # f0
        original_sources = data[2] # sources
        
        if network.F0Extractor is not None:
            # hcqt = data[3]  # hcqt
            # dphase = data[4]  # dphase
            # x, f0, original_sources, hcqt, dphase = x.to(device), f0.to(device), original_sources.to(device), hcqt.to(device), dphase.to(device)
            
            x, f0, original_sources = x.to(device), f0.to(device), original_sources.to(device)
        else:
            x, f0, original_sources = x.to(device), f0.to(device), original_sources.to(device)
        
        optimizer.zero_grad()
        
        if network.return_sources == True:
            if network.F0Extractor is not None:
                # test to show that if we use F0extractor, there is no need for the frequency
                f0 = torch.zeros_like(f0).to(device)
                
                if args.cuesta_model_trainable:                   
                    y_hat, sources, salience_maps, assignements = network(x, f0)

                    # Save Salience map to verify stuff
                    # for i in range(4):
                    #     plt.imshow(assignements[0, i].detach().cpu().numpy(), origin='lower', aspect='auto', cmap='magma')
                    #     plt.colorbar()
                    #     plt.savefig('./test_fig/salience_map_{}_torch.pdf'.format(i))
                    #     plt.close()
                        
                    #     plt.imshow(assignements[0, i].detach().cpu().numpy() * masks_batch[0,i].cpu().numpy(), origin='lower', aspect='auto', cmap='magma')
                    #     plt.colorbar()
                    #     plt.savefig('./test_fig/salience_map_{}_torch_mask.pdf'.format(i))
                    #     plt.close()
                    
                else:                    
                    # y_hat, sources = network(x, f0, hcqt, dphase)
                    y_hat, sources = network(x, f0)
            else:
                y_hat, sources = network(x, f0)
        else:
            y_hat = network(x, f0)

        loss = 0.
        if args.reconstruction_loss_weight > 0:
            loss_fn = losses.SpectralLoss(fft_sizes=args.loss_nfft,
                                          mag_weight=args.loss_mag_weight,
                                          logmag_weight=args.loss_logmag_weight,
                                          logmel_weight=args.loss_logmel_weight,
                                          delta_freq_weight=args.loss_delta_freq_weight,
                                          delta_time_weight=args.loss_delta_time_weight)
            if args.supervised:
                x = data[2].transpose(1, 2).reshape((args.batch_size * args.n_sources, -1)).to(device)  # true sources [batch_size * n_sources, n_samples]
                y_hat = y_hat[1].reshape((args.batch_size * args.n_sources, -1))  # source estimates [batch_size * n_sources, n_samples]
            reconstruction_loss = loss_fn(x, y_hat) * args.reconstruction_loss_weight
            loss += reconstruction_loss

        if args.ss_loss_weight > 0:
            ss_loss_fn = losses.SelfSupervisionLoss(ss_weights_dict)
            target_dict = data[2]
            ss_loss = ss_loss_fn(target_dict, y_hat) * args.ss_loss_weight
            loss += ss_loss

        if args.loss_lsf_weight > 0:
            lsf_loss_fn = losses.LSFRegularizer()
            y_hat, lsf = y_hat
            lsf_loss = lsf_loss_fn(lsf) * args.loss_lsf_weight
            loss -= lsf_loss
        
        if args.loss_saliences_weight > 0:
            loss_salience_fn = torch.nn.MSELoss()
            loss_salience = loss_salience_fn(salience_maps, assignements.sum(dim=1)[:,None,:,:]) * args.loss_saliences_weight
            loss += loss_salience
            
        if args.loss_voices_weight > 0:
            loss_voices_fn = torch.nn.MSELoss()
            loss_voices = loss_voices_fn(assignements * masks_batch, assignements) * args.loss_voices_weight
            loss += loss_voices
        
        loss.backward()
        optimizer.step()
        loss_container.update(loss.item(), f0.size(0))       
    
    # log audio to tensorboard
    if network.return_sources == True:
        # [batch_size, n_sources, n_samples]
        source_estimates_masking = utils.masking_from_synth_signals_torch(x, sources, n_fft=2048, n_hop=256)
        source_estimates_masking = source_estimates_masking.reshape((args.batch_size, args.n_sources, -1))
        
        writer.add_audio('train/mix/original', x[0] / torch.max(torch.abs(x[0])), global_step=epoch-1, sample_rate=args.samplerate)
        writer.add_audio('train/mix/reconstruct', y_hat[0] / torch.max(torch.abs(y_hat[0])), global_step=epoch-1, sample_rate=args.samplerate)
        
        for n_sources in range(args.n_sources):
            writer.add_audio(f'train/original_sources/source_{n_sources}', original_sources[0,:,n_sources] / torch.max(torch.abs(original_sources[0,:,n_sources])), global_step=epoch-1, sample_rate=args.samplerate)
            writer.add_audio(f'train/generated_sources/source_{n_sources}', sources[0][n_sources] / torch.max(torch.abs(sources[0][n_sources])), global_step=epoch-1, sample_rate=args.samplerate)
            writer.add_audio(f'train/mask_sources/source_{n_sources}', source_estimates_masking[0][n_sources] / torch.max(torch.abs(source_estimates_masking[0][n_sources])), global_step=epoch-1, sample_rate=args.samplerate)
    
    if args.loss_saliences_weight > 0:
        writer.add_scalar('Training_cost/loss_salience', loss_salience.item(), global_step=epoch-1)
    if args.loss_voices_weight > 0:
        writer.add_scalar('Training_cost/loss_voices', loss_voices.item(), global_step=epoch-1)
    
    return loss_container.avg


def valid(args, network, device, valid_sampler, epoch, writer):
    loss_container = utils.AverageMeter()
    network.eval()
    if args.supervised: network.return_sources = True
    
    if args.loss_voices_weight > 0:
        masks_batch = utils.get_training_masks(batch_mask_shape=(args.batch_size, args.n_sources, 360, 344), 
                                        freq_bands=[(179, 302), (152, 274), (119, 241), (77, 180)], 
                                        device=device)
    
    with torch.no_grad():
        for data in valid_sampler:
            x = data[0]  # audio
            f0 = data[1]  # f0
            original_sources = data[2] # sources
            
            if network.F0Extractor is not None:
                # hcqt = data[3]  # hcqt
                # dphase = data[4]  # dphase
                # x, f0, original_sources, hcqt, dphase = x.to(device), f0.to(device), original_sources.to(device), hcqt.to(device), dphase.to(device)
                
                x, f0, original_sources = x.to(device), f0.to(device), original_sources.to(device)
            else:
                x, f0, original_sources = x.to(device), f0.to(device), original_sources.to(device) #, z.to(device)
                

            if network.return_sources == True:                
                if network.F0Extractor is not None:
                    
                    # test to show that if we use F0extractor, there is no need for the frequency
                    f0 = torch.zeros_like(f0).to(device)
                    
                    if args.cuesta_model_trainable:
                        # y_hat, sources, salience_maps, salience_maps_reconstruct = network(x, f0)
                        y_hat, sources, salience_maps, assignements = network(x, f0)
                        
                    else:
                        # y_hat, sources = network(x, f0, hcqt, dphase)
                        # y_hat, sources, salience_maps, salience_maps_reconstruct  = network(x, f0)
                        y_hat, sources = network(x, f0)

                else:
                    y_hat, sources = network(x, f0)
            else:
                y_hat = network(x, f0)

            loss_fn = losses.SpectralLoss(fft_sizes=args.loss_nfft,
                                          mag_weight=args.loss_mag_weight,
                                          logmag_weight=args.loss_logmag_weight,
                                          logmel_weight=args.loss_logmel_weight,
                                          delta_freq_weight=args.loss_delta_freq_weight,
                                          delta_time_weight=args.loss_delta_time_weight)
            if args.supervised:
                batch_size = f0.size(0)
                x = data[2].transpose(1, 2).reshape((batch_size * args.n_sources, -1)).to(device)  # true sources [batch_size * n_sources, n_samples]
                y_hat = y_hat[1].reshape((batch_size * args.n_sources, -1))  # source estimates [batch_size * n_sources, n_samples]
            loss = loss_fn(x, y_hat)
            
            
            if args.loss_saliences_weight > 0:
                loss_salience_fn = torch.nn.MSELoss()
                loss_salience = loss_salience_fn(salience_maps, assignements.sum(dim=1)[:,None,:,:]) * args.loss_saliences_weight
                loss += loss_salience
            
            if args.loss_voices_weight > 0:
                loss_voices_fn = torch.nn.MSELoss()
                loss_voices = loss_voices_fn(assignements * masks_batch, assignements) * args.loss_voices_weight
                loss += loss_voices
                
            loss_container.update(loss.item(), f0.size(0))
        
        
        # end for loop
        # log audio to tensorboard
        if network.return_sources == True:
            batch_size = f0.size(0)
            
            # [batch_size * n_sources, n_samples]
            source_estimates_masking = utils.masking_from_synth_signals_torch(x, sources, n_fft=2048, n_hop=256)
            source_estimates_masking = source_estimates_masking.reshape((batch_size, args.n_sources, -1))
            writer.add_audio('valid/mix/original', x[0] / torch.max(torch.abs(x[0])), global_step=epoch-1, sample_rate=args.samplerate)
            writer.add_audio('valid/mix/reconstruct', y_hat[0] / torch.max(torch.abs(y_hat[0])), global_step=epoch-1, sample_rate=args.samplerate)
            
            for n_sources in range(args.n_sources):
                writer.add_audio(f'valid/original_sources/source_{n_sources}', original_sources[0,:,n_sources] / torch.max(torch.abs(original_sources[0,:,n_sources])), global_step=epoch-1, sample_rate=args.samplerate)
                writer.add_audio(f'valid/generated_sources/source_{n_sources}', sources[0][n_sources] / torch.max(torch.abs(sources[0][n_sources])), global_step=epoch-1, sample_rate=args.samplerate)
                writer.add_audio(f'valid/mask_sources/source_{n_sources}', source_estimates_masking[0][n_sources] / torch.max(torch.abs(source_estimates_masking[0][n_sources])), global_step=epoch-1, sample_rate=args.samplerate)    
        
        if args.loss_saliences_weight > 0:
            writer.add_scalar('Validation_cost/loss_salience', loss_salience.item(), global_step=epoch-1)
        if args.loss_voices_weight > 0:
            writer.add_scalar('Validation_cost/loss_voices', loss_voices.item(), global_step=epoch-1)
        
        return loss_container.avg


def get_statistics(args, dataset):

    # dataset is an instance of a torch.utils.data.Dataset class

    scaler = sklearn.preprocessing.StandardScaler()  # tool to compute mean and variance of data

    # define operation that computes magnitude spectrograms
    spec = torch.nn.Sequential(
        model.STFT(n_fft=args.nfft, n_hop=args.nhop),
        model.Spectrogram(mono=True)
    )
    # return a deep copy of dataset:
    # constructs a new compound object and recursively inserts copies of the objects found in the original
    dataset_scaler = copy.deepcopy(dataset)

    dataset_scaler.samples_per_track = 1
    dataset_scaler.augmentations = None  # no scaling of sources before mixing
    dataset_scaler.random_chunks = False  # no random chunking of tracks
    dataset_scaler.random_track_mix = False  # no random accompaniments for vocals
    dataset_scaler.random_interferer_mix = False
    dataset_scaler.seq_duration = None  # if None, the original whole track from musdb is loaded

    # make a progress bar:
    # returns an iterator which acts exactly like the original iterable,
    # but prints a dynamically updating progressbar every time a value is requested.
    pbar = tqdm.tqdm(range(len(dataset_scaler)), disable=args.quiet)

    for ind in pbar:
        out = dataset_scaler[ind]  # x is mix and y is target source in time domain, z is text and ignored here
        x = out[0]
        y = out[1]
        pbar.set_description("Compute dataset statistics")
        X = spec(x[None, ...])  # X is mono magnitude spectrogram, ... means as many ':' as needed

        # X is spectrogram of one full track
        # at this point, X has shape (nb_frames, nb_samples, nb_channels, nb_bins) = (N, 1, 1, F)
        # nb_frames: time steps, nb_bins: frequency bands, nb_samples: batch size

        # online computation of mean and std on X for later scaling
        # after squeezing, X has shape (N, F)
        scaler.partial_fit(np.squeeze(X))  # np.squeeze: remove single-dimensional entries from the shape of an array

    # set inital input scaler values
    # scale_ and mean_ have shape (nb_bins,), standard deviation and mean are computed on each frequency band separately
    # if std of a frequency bin is smaller than m = 1e-4 * (max std of all freq. bins), set it to m
    std = np.maximum(   # maximum compares two arrays element wise and returns the maximum element wise
        scaler.scale_,
        1e-4*np.max(scaler.scale_)  # np.max = np.amax, it returns the max element of one array
    )
    return scaler.mean_, std


def main():
    parser = configargparse.ArgParser()
    parser.add('-c', '--my-config', required=False, is_config_file=True, help='config file path', default='config.txt')
    #parser = argparse.ArgumentParser(description='Training')

    # experiment tag which will determine output folder in trained models, tensorboard name, etc.
    parser.add_argument('--tag', type=str, default='test')

    # allow to pass a comment about the experiment
    parser.add_argument('--comment', type=str, help='comment about the experiment')

    args, _ = parser.parse_known_args()

    # Dataset paramaters
    parser.add_argument('--dataset', type=str, default="musdb",
                        help='Name of the dataset.')
    parser.add_argument('--one-example', action='store_true', default=False,
                        help='overfit to one example of the training set')
    parser.add_argument('--one-batch', action='store_true', default=False,
                        help='overfit to one batch of the training set')
    parser.add_argument('--parallel', action='store_true', default=False,
                        help='if True, correlated sources are generated in parallel in synth. dataset')
    parser.add_argument('--one-song', action='store_true', default=False,
                        help='if True, only one song is used in BC dataset for training and validation')
    parser.add_argument('--cuesta-model', action='store_true', default=False,
                        help='if True, use cuesta model inside the network')



    parser.add_argument('--output', type=str, default="trained_models/{}/".format(args.tag),
                        help='provide output path base folder name')

    parser.add_argument('--wst-model', type=str, help='Path to checkpoint folder for warmstart')

    # Training Parameters
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate, defaults to 1e-3')
    parser.add_argument('--patience', type=int, default=140,
                        help='maximum number of epochs to train (default: 140)')
    parser.add_argument('--lr-decay-patience', type=int, default=80,
                        help='lr decay patience for plateau scheduler')
    parser.add_argument('--lr-decay-gamma', type=float, default=0.3,
                        help='gamma of learning rate scheduler decay')
    parser.add_argument('--weight-decay', type=float, default=0.00001,
                        help='weight decay')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--supervised', action='store_true', default=False)


    parser.add_argument('--reconstruction-loss-weight', type=float, default=1.0)
    parser.add_argument('--loss-nfft', type=int, action='store', nargs='*', default=[2048, 1024, 512, 256, 128, 64])
    parser.add_argument('--loss-mag-weight', type=float, default=1.0)
    parser.add_argument('--loss-logmag-weight', type=float, default=1.0)
    parser.add_argument('--loss-logmel-weight', type=float, default=0.0)
    parser.add_argument('--loss-delta-freq-weight', type=float, default=0.0)
    parser.add_argument('--loss-delta-time-weight', type=float, default=0.0)
    parser.add_argument('--loss-lsf-weight', type=float, default=0.0)
    parser.add_argument('--ss-loss-weight', type=float, default=0.0)
    parser.add_argument('--harmonic-amp-loss-weight', type=float, default=0.0)
    parser.add_argument('--f0-hz-loss-weight', type=float, default=0.0)
    parser.add_argument('--harmonics-roll-off-loss-weight', type=float, default=0.0)
    parser.add_argument('--lsf-loss-weight', type=float, default=0.0)
    parser.add_argument('--noise-gain-loss-weight', type=float, default=0.0)
    parser.add_argument('--noise-mags-loss-weight', type=float, default=0.0)
    parser.add_argument('--loss-saliences-weight', type=float, default=0.0)
    parser.add_argument('--loss-voices-weight', type=float, default=0.0)



    # Model Parameters
    parser.add_argument('--nfft', type=int, default=512,
                        help='STFT fft size and window size')
    parser.add_argument('--nhop', type=int, default=256,
                        help='STFT hop size')
    parser.add_argument('--filter-order', type=int, default=10,
                        help='filter order of vocal tract all-pole filter')

    parser.add_argument('--noise-filter-mags', type=int, default=40,
                        help='number of frequency bands in noise filter')
    parser.add_argument('--encoder', type=str, default='SeparationEncoderSimple')
    parser.add_argument('--encoder-hidden-size', type=int, default=256)
    parser.add_argument('--embedding-size', type=int, default=128)
    parser.add_argument('--decoder-hidden-size', type=int, default=512)
    parser.add_argument('--decoder-output-size', type=int, default=512)
    parser.add_argument('--n-sources', type=int, default=2)
    parser.add_argument('--estimate-lsf', action='store_true', default=False)
    parser.add_argument('--estimate-noise-mags', action='store_true', default=False)
    parser.add_argument('--unidirectional', action='store_true', default=False)
    parser.add_argument('--voiced-unvoiced-same-noise', action='store_true', default=False)
    parser.add_argument('--return-sources', action='store_true', default=False)
    parser.add_argument('--cuesta-model-trainable', action='store_true', default=False,
                        help='if True, cuesta model is trainable')


    parser.add_argument('--nb-workers', type=int, default=4,
                        help='Number of workers for dataloader.')

    # name of the model class in model.py that should be used
    parser.add_argument('--architecture', type=str)
    parser.add_argument('--nb-filter-magnitudes', type=int, default=65)
    parser.add_argument('--estimate-f0', action='store_true', default=False)
    parser.add_argument('--supervised-f0', action='store_true', default=False)
    parser.add_argument('--switch-off-noise', action='store_true', default=False)
    parser.add_argument('--f-ref-source-spec', type=float, default=200.)
    parser.add_argument('--harmonic-roll-off', type=float, default=12.)
    parser.add_argument('--source-spectrum', type=str, default='flat')
    parser.add_argument('--original-cu-net', action='store_true', default=False)

    # Misc Parameters
    parser.add_argument('--quiet', action='store_true', default=False,
                        help='less verbose during training')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')


    args, _ = parser.parse_known_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print("Using GPU:", use_cuda)
    #print("Using Torchaudio: ", utils._torchaudio_available())
    print('Cuesta model:', args.cuesta_model)
    dataloader_kwargs = {'num_workers': args.nb_workers, 'pin_memory': True} if use_cuda else {}

    # create output dir if not exist
    target_path = Path(args.output)
    target_path.mkdir(parents=True, exist_ok=True)

    # copy config.txt to output dir
    shutil.copy2('config.txt', target_path)

    writer = SummaryWriter(log_dir=os.path.join('tensorboard', args.tag))

    # use jpg or npy
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if use_cuda else "cpu")

    train_dataset, valid_dataset, args = data.load_datasets(parser, args)


    train_sampler = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        worker_init_fn=utils.worker_init_fn,
        **dataloader_kwargs
    )
    valid_sampler = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, drop_last=True, **dataloader_kwargs
    )

    # make dict for self supervision loss weights
    ss_weights_dict = {'harmonic_amplitudes': args.harmonic_amp_loss_weight,
                       'harmonic_distribution': 0.,
                       'f0_hz': args.f0_hz_loss_weight,
                       'harmonics_roll_off': args.harmonics_roll_off_loss_weight,
                       'line_spectral_frequencies': args.lsf_loss_weight,
                       'noise_gain': args.noise_gain_loss_weight,
                       'voiced_unvoiced': 0.,
                       'voiced_noise_magnitudes': args.noise_mags_loss_weight,
                       }


    train_args_dict = vars(args)

    train_params_dict = copy.deepcopy(vars(args))  # return args as dictionary with no influence on args

    model_class = model_utls.ModelLoader.get_model(args.architecture)
    model_to_train = model_class.from_config(train_params_dict)
    
    if args.cuesta_model:
        # Si True, on utilise le modèle Cuesta entrainé
        # Sinon, on utilise le modèle Cuesta non entrainé
        model_to_train.F0Extractor = models.F0Extractor(trained_cuesta=True) # ATTENTION: Pour l'instant trained_cuesta est un paramètre en dur
        model_to_train.F0Assigner = models.Assigner(trained_VA=True)
        
        if args.cuesta_model_trainable:
            model_to_train.cuesta_model_trainable = args.cuesta_model_trainable
            model_to_train.F0Extractor = model_to_train.F0Extractor.train()
            model_to_train.F0Assigner = model_to_train.F0Assigner.train()
        else:
            model_to_train.F0Extractor = model_to_train.F0Extractor.eval()
            model_to_train.F0Assigner = model_to_train.F0Assigner.eval()
            
        print('Cuesta_trainable:', model_to_train.cuesta_model_trainable)
        
    model_to_train.to(device)

    optimizer = torch.optim.Adam(
        model_to_train.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 200, gamma=args.lr_decay_gamma)

    es = utils.EarlyStopping(patience=args.patience)

    # if a model is specified: resume training
    if args.wst_model:
        model_path = Path(os.path.join('trained_models', args.wst_model)).expanduser()
        with open(Path(model_path, args.wst_model + '.json'), 'r') as stream:
            results = json.load(stream)

        target_model_path = Path(model_path, args.wst_model + ".pth")
        checkpoint_path = Path(model_path, args.wst_model + ".chkpnt")
        
        # Load juste the weights of the model
        state_dict = torch.load(target_model_path, map_location=device)
        
        # Load all the informations of the model - optimizer, scheduler, etc.
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model_to_train.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])              
        
        # train for another epochs_trained
        t = tqdm.trange(
            results['epochs_trained'],
            results['epochs_trained'] + args.epochs + 1,
            disable=args.quiet
        )
        train_losses = results['train_loss_history']
        valid_losses = results['valid_loss_history']
        train_times = results['train_time_history']
        # we don't set the parameters below to allow resuming training on different data set
        # (model is saved with new name, so there is no risk of overwriting)
        best_epoch = 0

    # else start from 0
    else:
        t = tqdm.trange(1, args.epochs + 1, disable=args.quiet)
        train_losses = []
        valid_losses = []
        train_times = []
        best_epoch = 0

    for epoch in t:
        t.set_description("Training Epoch")
        end = time.time()
        
        train_loss = train(args, model_to_train, device, train_sampler, optimizer, ss_weights_dict, epoch, writer)

        # calculate validation loss only if model is not optimized on one single example
        if args.one_example or args.one_batch or (args.dataset == 'synthetic'):
            # if overfitting on one example, early stopping is done based on training loss
            valid_loss = train_loss
        else:
            valid_loss = valid(args, model_to_train, device, valid_sampler, epoch, writer)
            writer.add_scalar("Validation_cost", valid_loss, epoch)
            valid_losses.append(valid_loss)

        writer.add_scalar("Training_cost", train_loss, epoch)

        scheduler.step()
        train_losses.append(train_loss)

        stop = es.step(valid_loss)

        if valid_loss == es.best:
            best_epoch = epoch

        t.set_postfix(
            train_loss=train_loss, val_loss=valid_loss
        )

        utils.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model_to_train.state_dict(),
            'best_loss': es.best,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        },
            is_best=valid_loss == es.best,
            path=target_path,
            tag=args.tag
        )

        # save params
        params = {
            'epochs_trained': epoch,
            'args': vars(args),
            'best_loss': es.best,
            'best_epoch': best_epoch,
            'train_loss_history': train_losses,
            'valid_loss_history': valid_losses,
            'train_time_history': train_times,
            'num_bad_epochs': es.num_bad_epochs
        }

        with open(Path(target_path,  args.tag + '.json'), 'w') as outfile:
            outfile.write(json.dumps(params, indent=4, sort_keys=True))

        train_times.append(time.time() - end)

        if stop:
            print("Apply Early Stopping")
            break



def bkld(y_pred, y_true):
    
    epsilon = 1e-7
    y_true = torch.clamp(y_true, epsilon, 1.0 - epsilon)
    y_pred = torch.clamp(y_pred, epsilon, 1.0 - epsilon)
    
    bce_loss = torch.nn.BCELoss(reduction='mean')
    
    return bce_loss(y_pred, y_true)



def cuesta_train(model, train_loader, n_epoch, device):
    """ Training loop for Cuesta model
    
    => MANQUE l'EARLY STOPPING pour être conforme à l'article

    Args:
        model (_type_): _description_
        train_loader (_type_): _description_
        n_epoch (_type_): _description_
        device (_type_): _description_
    """
    
    acc_mse = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(n_epoch):
        
        train_loss = 0.0
        train_acc = 0.0
        
        for i, data in enumerate(train_loader, 0):
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            
            print(inputs[:,1,:,:].is_cuda)
            
            optimizer.zero_grad()
            
            preds = model(inputs[:,0,:,:], inputs[:,1,:,:])
            bce_loss = bkld(preds, targets)
            acc = acc_mse(preds, targets)
            
            bce_loss.backward()
            optimizer.step()
            
            train_loss += bce_loss.item()
            train_acc += acc.item()
    
        # calculating the average training and validation loss over epoch
        epoch_loss = train_loss / len(train_loader)
        
        # printing average training and average validation losses
        print("Epoch: {}".format(epoch+1))
        print("Training loss: {:.4f}".format(epoch_loss))
        print("Training accuracy: {:.4f}".format(train_acc / len(train_loader)))
        
        
        

def cuesta_unit_test():    
    # create model
    cuesta_model = models.F0Extractor(trained_cuesta=False)
    
    use_cuda = False
    device = torch.device("cuda" if use_cuda else "cpu")
    
    audio_fpath = "/home/pchouteau/umss/umss/Datasets/BC/mixtures_2_sources/1_BC001_part1_ab.wav"
    pump = data.create_pump_object()
    features = data.compute_pump_features(pump, audio_fpath)

    hcqt = features['dphase/mag'][0]
    dphase = features['dphase/dphase'][0]
    
    hcqt = hcqt.transpose(2, 1, 0)[np.newaxis, :, :, :]
    dphase = dphase.transpose(2, 1, 0)[np.newaxis, :, :, :]
    
    
    inps = torch.stack((torch.from_numpy(hcqt[:, :, :, :50]), torch.from_numpy(dphase[:, :, :, :50])), dim=1)
    print(inps.shape)
    tgts = torch.rand(1, 360, 50, requires_grad=False)
    
    dataset = torch.utils.data.TensorDataset(inps, tgts)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=0
    )
    
    cuesta_train(cuesta_model, train_loader, 100, device)
    


if __name__ == "__main__":
    main()
    

    # import models 
    # cuesta_unit_test()