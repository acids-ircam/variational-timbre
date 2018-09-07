#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 17:54:24 2018

@author: chemla
"""
import torch, os, pdb, copy
from utils.dataloader import DataLoader

import visualize.visualize_plotting as lplt
from visualize.visualize_dimred import PCA

import torchvision
import numpy as np

import matplotlib.pyplot as plt


def update_losses(losses_dict, new_losses):
    for k, v in new_losses.items():
        if not k in losses_dict.keys():
            losses_dict[k] = []
        losses_dict[k].append(new_losses[k])
    return losses_dict


def train_model(dataset, model, loss, task=None, loss_task=None, options={}, plot_options={}, save_with={}):  
    # Global training parameters
    name = options.get('name', 'model')
    epochs = options.get('epochs', 10000)
    save_epochs = options.get('save_epochs', 2000)
    best_save_epochs = options.get('best_save_epochs', save_epochs)
    plot_epochs = options.get('plot_epochs', 100)
    batch_size = options.get('batch_size', 64)
    image_export = options.get('image_export', False)
    nb_reconstructions = options.get('nb_reconstructions', 3)
    save_threshold = options.get('save_threshold', 100)
    remote = options.get('remote', None)
    if loss_task is None:
        loss_task = task if not task is None else None
    
    # Setting results & plotting directories
    results_folder = options.get('results_folder', 'saves/'+name)
    figures_folder = options.get('figures_folder', results_folder+'/figures')
    if not os.path.isdir(results_folder):
        os.makedirs(results_folder)
    if not os.path.isdir(figures_folder):
        os.makedirs(figures_folder)
        
    # Init training
    epoch = -1 
    min_test_loss = np.inf; best_model = None
    reconstruction_ids = np.random.permutation(len(dataset))[:nb_reconstructions**2]
    best_model = None
        
    # Start training!
    while epoch < epochs:
        print('-----EPOCH %d'%epoch)
        epoch += 1
        loader = DataLoader(dataset, batch_size=batch_size, partition='train', task=task)
        
        # train phase
        batch = 0; current_loss = 0;
        train_losses = None
        model.train()
        for x,y in loader:
            x = model.format_input_data(x); y = model.format_label_data(y);
            if loss_task == task:
                y_task = y 
            else:
                if not loss_task is None:
                    plabel = {'dim':dataset.classes[loss_task]['_length']}
                    y_task = model.format_label_data(dataset.metadata.get(loss_task)[loader.current_ids], plabel=plabel)
                else:
                    y_task = None
            out = model.forward(x, y=y)
            batch_loss, losses = loss.loss(model, out, x=x, y=y_task, epoch=epoch)
            if train_losses is None:
                train_losses = losses 
            else:
                train_losses += losses 
            model.step(batch_loss)
            print("epoch %d / batch %d / losses : %s "%(epoch, batch, loss.get_named_losses(losses)))
            current_loss += batch_loss
            batch += 1
        current_loss /= batch
        print('--- FINAL LOSS : %s'%current_loss)
        loss.write('train', train_losses)
        
        ## test_phase
        with torch.no_grad():
            model.eval()
            test_data = model.format_input_data(dataset['test'][:])
            test_metadata = model.format_label_data(dataset.metadata[task][dataset.partitions['test']]) if not task is None else None
            out = model.forward(test_data, y=test_metadata)
            if loss_task == task:
                y_task = y 
            else:
                if not loss_task is None:
                    plabel = {'dim':dataset.classes[loss_task]['_length']}
                    test_ids = dataset.partitions.get('test')
                    y_task = model.format_label_data(dataset.metadata.get(loss_task)[test_ids], plabel=plabel)
                else:
                    y_task = None
            test_loss, losses = loss.loss(model, out, x=test_data, y=y_task)
            loss.write('test', losses)
            if test_loss < min_test_loss and epoch > save_threshold:
                min_test_loss = test_loss
                print('-- saving best model at %s'%'results/%s/%s_%d.t7'%(results_folder, name, epoch))
                model.save('%s/%s_best.t7'%(results_folder, name), loss=loss, epoch=epoch, partitions=dataset.partitions)
            model.schedule(test_loss)
        
        plt.ioff()
        
        # Save models
        if epoch%save_epochs==0:
            print('-- saving model at %s'%'results/%s/%s_%d.t7'%(results_folder, name, epoch))
            model.save('%s/%s_%d.t7'%(results_folder, name, epoch), loss=loss, epoch=epoch, partitions=dataset.partitions, **save_with)

        # Make plots
        if epoch%plot_epochs == 0:
            plt.close('all')
            n_points = plot_options.get('plot_npoints', min(dataset.data.shape[0], 5000))
            plot_tasks = plot_options.get('plot_tasks', dataset.tasks)
            plot_dimensions = plot_options.get('plot_dimensions', list(range(model.platent[-1]['dim'])))
            plot_layers = plot_options.get('plot_layers', list(range(len(model.platent))))
            if plot_options.get('plot_reconstructions', True):
                print('plotting reconstructions...')
                lplt.plot_reconstructions(dataset, model, label=task, out=figures_folder+'/reconstructions_%d.svg'%epoch)
            if plot_options.get('plot_latentspace', True):
                transformation = PCA(n_components=3)
                print('plotting latent spaces...')
                lplt.plot_latent3(dataset, model, transformation, label=task, tasks=plot_tasks, layers=plot_layers, n_points=n_points, out=figures_folder+'/latent_%d'%epoch)
            if plot_options.get('plot_statistics', True):
                print('plotting latent statistics...')
                lplt.plot_latent_stats(dataset, model, label=task, tasks=plot_tasks, layers=plot_layers, legend=True, n_points=n_points, balanced=True, out=figures_folder+'/statistics_%d'%epoch)
            if plot_options.get('plot_distributions', True):
                print('plotting latent distributions...')
                lplt.plot_latent_dists(dataset, model, label=task, tasks=plot_tasks, n_points=n_points, out=figures_folder+'/dists_%d'%epoch, 
                                       dims=plot_dimensions,split=False, legend=True, bins=10, relief=True)
            if plot_options.get('plot_losses', True):
                print('plotting losses...')
                lplt.plot_class_losses(dataset, model, loss, label=task, tasks=plot_tasks, loss_task = loss_task, out=figures_folder+'/losses')

            if not remote is None:
                print('scp -r %s %s:'%(figures_folder, remote))
                os.system('scp -r %s %s:'%(figures_folder, remote))
                
        if image_export:
            images = dataset[reconstruction_ids]
            if not task is None:
                metadata = dataset.metadata[task][reconstruction_ids]
            else:
                metadata = None
            out = model.pinput[0]['dist'](*model.forward(images, y=metadata)['x_params'][0]).mean
            torchvision.utils.save_image(out.reshape(out.size(0), 1, 28, 28), figures_folder+'grid_%d.png'%epoch, nrow=nb_reconstructions)
        
    model.save('%s/%s_final.t7'%(results_folder, name), loss=loss, epoch=epoch, partitions=dataset.partitions, **save_with)


        
