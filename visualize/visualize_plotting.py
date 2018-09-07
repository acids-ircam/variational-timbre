import torch, os, pdb
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce

from mpl_toolkits.mplot3d import Axes3D
import visualize.visualize_dimred as dr
from utils.onehot import fromOneHot
import matplotlib.patches as mpatches



################################################
########        UTILS
####

def get_cmap(n, color_map='plasma'):
    return plt.cm.get_cmap(color_map, n)

def get_class_ids(dataset, task, ids=None):
    if ids is None:
        ids = np.arange(dataset.data.shape[0])
    metadata = np.array(dataset.metadata.get(task)[ids])
    if metadata is None:
        raise Exception('please give a valid metadata task')
    n_classes = list(set(metadata))
    ids = []
    for i in n_classes:
        ids.append(np.where(metadata==i)[0])
    return ids, n_classes

def get_divs(n):
    primfac = []
    d = 2
    while d*d <= n:
        while (n % d) == 0:
            primfac.append(d)  # supposing you want multiple factors repeated
            n //= d
        d += 1
    if n > 1:
       primfac.append(n)
    primfac = np.array(primfac)
    return np.prod(primfac[0::2]), np.prod(primfac[1::2])


def get_balanced_ids(metadata, n_points=None):
    items = list(set(metadata))
    class_ids = []
    n_ids = metadata.shape[0] // len(items) if n_points is None else int(n_points) // len(items)
    for i, item in enumerate(items):
        class_ids.append(np.where(metadata==item)[0])
        if len(class_ids[-1]) < (n_ids // len(items)):
            n_ids = len(class_ids[-1])
    class_ids = [list(i[:n_ids]) for i in class_ids]
    ids = reduce(lambda x,y: x+y, class_ids)
    return ids


################################################
########        LATENT PLOTS
####

def plot_latent2(dataset, model, transformation, n_points=None, tasks=None, classes=None, label=None, balanced=False, legend=True,
                   sample = False, layers=0, color_map="plasma", zoom=10, out=None, verbose=False, *args, **kwargs):
    # select points
    if balanced and tasks!=None:
        ids = set()
        task_ids = []
#        pdb.set_trace()
        for t in tasks:
            task_ids.append(get_balanced_ids(dataset.metadata[t], n_points))
            ids = ids.union(set(task_ids[-1]))
        ids = list(ids)
        task_ids = [np.array([ids.index(x) for x in task_id]) for task_id in task_ids]
    else:
        ids = dataset.data.shape[0] if n_points is None else np.random.permutation(dataset.data.shape[0])[:n_points]
        task_ids = [] if tasks is None else [range(len(ids))]*len(tasks)
        
    ids = np.array(ids)
    
    data = model.format_input_data(dataset.data[ids])
    metadata = model.format_label_data(dataset.metadata[label][ids]) if not label is None else None
    output = model.forward(data, y=metadata, *args, **kwargs)
    figs = []
    
    for layer in layers:
        # make manifold
        
        if tasks is None:
            fig = plt.figure('latent plot of layer %d'%layer)
            
            current_z = model.platent[layer]['dist'](*output['z_params_enc'][layer]).mean.detach().numpy()
            if current_z.shape(1) > 2:
                current_z = transformation.fit_transform(current_z)
                
            plt.scatter(output['z_params_enc'][layer][:, 0], output['z_params_enc'][layer][:,1])
            plt.title('latent plot of layer %d'%layer)
            if not out is None:
                fig.savefig(out+'_layer%d.svg'%layer, format="svg")
            figs.append(fig)
        else:
            if not issubclass(type(tasks), list):
                tasks = [tasks]
            current_z = model.platent[layer]['dist'](*output['z_params_enc'][layer]).mean.detach().numpy()
            if current_z.shape[1] > 2:
                current_z = transformation.fit_transform(current_z)
            for id_task, task in enumerate(tasks):
                print('-- plotting task %s'%task)
                fig = plt.figure('latent plot of layer %d, task %s'%(layer, task))
    #            pdb.set_trace()
                meta = np.array(dataset.metadata[task])[ids[task_ids[id_task]]]
                _, classes = get_class_ids(dataset, task, ids=ids[task_ids[id_task]])
    #            pdb.set_trace()
                
                cmap = get_cmap(len(classes))
                current_z = current_z[task_ids[id_task], :] if balanced else current_z
                plt.scatter(current_z[:, 0], current_z[:,1], c=cmap(meta))
                if legend:
                    handles = []
                    if dataset.classes.get(task)!=None:
                        class_names = {v: k for k, v in dataset.classes[task].items()}
                        for cl in classes:
                            patch = mpatches.Patch(color=cmap(cl), label=class_names[cl])
                            handles.append(patch)
                        fig.legend(handles=handles)
                        
                figs.append(fig)
                if not out is None:
                    title = out+'_layer%d_%s.svg'%(layer, task)
                    fig.savefig(title, format="svg")
    return figs

def plot_latent3(dataset, model, transformation, n_points=None, label=None, tasks=None, classes=None, balanced=False, 
                   sample = False, layers=[0], color_map="plasma", zoom=10, out=None, legend=True, centroids=False, *args, **kwargs):
    # select points
    # make manifold
    if balanced and tasks!=None:
        ids = set()
        task_ids = []
#        pdb.set_trace()
        for t in tasks:
            task_ids.append(get_balanced_ids(dataset.metadata[t], n_points))
            ids = ids.union(set(task_ids[-1]))
        ids = list(ids)
        task_ids = [np.array([ids.index(x) for x in task_id]) for task_id in task_ids]
    else:
        ids = dataset.data.shape[0] if n_points is None else np.random.permutation(dataset.data.shape[0])[:n_points]
        task_ids = [] if tasks is None else [range(len(ids))]*len(tasks)
        
    ids = np.array(ids)
    
    data = model.format_input_data(dataset.data[ids])
    metadata = model.format_label_data(dataset.metadata[label][ids]) if not label is None else None
    output = model.forward(data, y=metadata, *args, **kwargs)
    figs = []
    
    for layer in layers:
        # make manifold
        
        if tasks is None:
            fig = plt.figure('latent plot of layer %d'%layer)
            ax = fig.gca(projection='3d')
            
            current_z = model.platent[layer]['dist'](*output['z_params_enc'][layer]).mean.detach().cpu().numpy()
            if current_z.shape(1) > 3:
                current_z = transformation.fit_transform(current_z)
                
            ax.scatter(output['z_params_enc'][layer][:, 0], output['z_params_enc'][layer][:,1], output['z_params_enc'][layer][:, 2])
            plt.title('latent plot of layer %d'%layer)
            if not out is None:
                fig.savefig(out+'_layer%d.svg'%layer, format="svg")
            figs.append(fig)
        else:
            if not issubclass(type(tasks), list):
                tasks = [tasks]
            current_z = model.platent[layer]['dist'](*output['z_params_enc'][layer]).mean.cpu().detach().numpy()
            if current_z.shape[1] > 3:
                current_z = transformation.fit_transform(current_z)
            for id_task, task in enumerate(tasks):
                print('-- plotting task %s'%task)
                fig = plt.figure('latent plot of layer %d, task %s'%(layer, task))
                ax = fig.gca(projection='3d')
                meta = np.array(dataset.metadata[task])[ids[task_ids[id_task]]]
                class_ids, classes = get_class_ids(dataset, task, ids=ids[task_ids[id_task]])
                
                cmap = get_cmap(len(classes))
                current_z = current_z[task_ids[id_task], :] if balanced else current_z
                
                if centroids:
                    current_alpha = 0.06
                else:
                    current_alpha = 0.7
                
                if current_z.shape[1]==2:
                    ax.scatter(current_z[:, 0], current_z[:,1], 0, c=cmap(meta), alpha = current_alpha)
                else:
                    ax.scatter(current_z[:, 0], current_z[:,1], current_z[:, 2], c=cmap(meta), alpha = current_alpha)
                    
                class_names = {v: k for k, v in dataset.classes[task].items()}
                if centroids:
                    for i, cid in enumerate(class_ids):
                        centroid = np.mean(current_z[cid], axis=0)
                        ax.scatter(centroid[0], centroid[1], centroid[2], s = 30, c=cmap(classes[i]))
                        ax.text(centroid[0], centroid[1], centroid[2], class_names[i], color=cmap(classes[i]), fontsize=10)
                    
                if legend:
                    handles = []
                    if dataset.classes.get(task)!=None:
                        for cl in classes:
                            patch = mpatches.Patch(color=cmap(cl), label=class_names[cl])
                            handles.append(patch)
                        fig.legend(handles=handles)
                        
                figs.append(fig)
                if not out is None:
                    title = out+'_layer%d_%s.svg'%(layer, task)
                    fig.savefig(title, format="svg")
    return figs



def plot_latent_stats(dataset, model, label=None, tasks=None, n_points=None, layers=[0], legend=True, out=None, balanced=False):
    
    if balanced and tasks!=None:
        ids = set()
        task_ids = []
#        pdb.set_trace()
        for t in tasks:
            task_ids.append(get_balanced_ids(dataset.metadata[t], n_points))
            ids = ids.union(set(task_ids[-1]))
        ids = list(ids)
        task_ids = [np.array([ids.index(x) for x in task_id]) for task_id in task_ids]
    else:
        ids = dataset.data.shape[0] if n_points is None else np.random.permutation(dataset.data.shape[0])[:n_points]
        task_ids = [] if tasks is None else [range(len(ids))]*len(tasks)
        
    ids = np.array(ids)
        
    data = model.format_input_data(dataset.data[ids])
    y = model.format_label_data(dataset.metadata.get(label))
    if not y is None:
        y = y[ids]
        
    vae_out = model.encode(data, y=y)    
    
    figs = []
    for layer in layers:
        zs = [x.cpu().detach().cpu().numpy() for x in vae_out[0][layer]]
        latent_dim = zs[0].shape[1]
        id_range = np.array(list(range(latent_dim)))
        if tasks is None:
            fig = plt.figure('latent statistics for layer %d'%layer)
            ax1 = fig.add_subplot(211); ax1.set_title('variance of latent positions')
            ax2 = fig.add_subplot(212); ax2.set_title('mean of variances per axis')
            pos_var = [np.std(zs[0], 0)]
            var_mean = [np.mean(zs[1], 0)]
            width = 1/len(pos_var)
            cmap = get_cmap(len(pos_var))
            for i in range(len(pos_var)):
                ax1.bar(id_range+i*width, pos_var[i], width)
                ax2.bar(id_range+i*width, var_mean[i], width)
    #        ax1.set_xticklabels(np.arange(latent_dim), np.arange(latent_dim))
    #        ax2.set_xticklabels(np.arange(latent_dim), np.arange(latent_dim))
            if not out is None:
                fig.savefig(out+'_layer%d.svg'%layer, format="svg")
            figs.append(fig)
        else:
            if not issubclass(type(tasks), list):
                tasks = [tasks]
            for t, task in enumerate(tasks):
                print('-- plotting task %s'%task)
                fig = plt.figure('latent statistics for layer %d, task %s'%(layer, task))
                ax1 = fig.add_subplot(211); ax1.set_title('variance of latent positions')
                ax2 = fig.add_subplot(212); ax2.set_title('mean of variances per axis')
                # get classes
                class_ids, classes = get_class_ids(dataset, task, ids=ids[task_ids[t]])
                # get data
                pos_var = []; var_mean= [];
                width = 1/len(class_ids)
                cmap = get_cmap(len(class_ids))
                handles = []
                for i, c in enumerate(class_ids):
                    pos_var.append(np.std(zs[0][class_ids[i]], 0))
                    var_mean.append(np.mean(zs[1][class_ids[i]], 0))
                    ax1.bar(id_range+i*width, pos_var[i], width, color=cmap(i))
                    ax2.bar(id_range+i*width, var_mean[i], width, color=cmap(i))
                if legend:
                    handles = []
                    class_names = {v: k for k, v in dataset.classes[task].items()}
                    for i in classes:
                        patch = mpatches.Patch(color=cmap(i), label=class_names[i])
                        handles.append(patch)
                    fig.legend(handles=handles)
                if not out is None:
                    title = out+'_layer%d_%s.svg'%(layer, task)
                    fig.savefig(title, format="svg")
                figs.append(fig)
            
                
    # plot histograms
    return figs
        

def plot_latent_dists(dataset, model, label=None, tasks=None, bins=20, layers=[0], n_points=None, dims=None, legend=True, split=False, out=None, relief=True, ids=None, **kwargs):
    # get data ids
    if n_points is None:
        ids = np.arange(dataset.data.shape[0]) if ids is None else ids
        data = dataset.data
        y = dataset.metadata.get(label)
    else:
        ids = np.random.permutation(dataset.data.shape[0])[:n_points] if ids is None else ids
        data = dataset.data[ids]
        y = dataset.metadata.get(label)
        if not y is None:
            y = y[ids]
    y = model.format_label_data(y)
    data = model.format_input_data(data);
    
    if dims is None:
        dims = list(range(model.platent[layer]['dim']))
        
    # get latent space
    with torch.no_grad():
        vae_out = model.encode(data, y=y)
        # get latent means of corresponding parameters
    
    # get  
    figs = []
    
    for layer in layers:
        zs = model.platent[layer]['dist'](*vae_out[0][layer]).mean.cpu().detach().numpy()
        if split:
            if tasks is None:
                for dim in dims:
                    fig = plt.figure('dim %d'%dim, figsize=(20,10))
                    hist, edges = np.histogram(zs[:, dim], bins=bins)
                    plt.bar(edges[:-1], hist, edges[1:] - edges[:-1], align='edge')
                    if not out is None:
                        prefix = out.split('/')[:-1]
                        fig.savefig(prefix+'/dists/'+out.split('/')[-1]+'_%d_dim%d.svg'%(layer, dim))
                    figs.append(fig)
            else:
                if not os.path.isdir(out+'/dists'):
                    os.makedirs(out+'/dists')
                for t in range(len(tasks)):
                    class_ids, classes = get_class_ids(dataset, tasks[t], ids=ids)
                    cmap = get_cmap(len(class_ids))
                    for dim in dims:
                        fig = plt.figure('dim %d'%dim, figsize=(20, 10))
                        ax = fig.gca(projection='3d') if relief else fig.gca()
                        for k, cl in enumerate(class_ids):
                            hist, edges = np.histogram(zs[cl, dim], bins=bins)
                            colors = cmap(k)
                            if relief:
                                ax.bar3d(edges[:-1], k*np.ones_like(hist), np.zeros_like(hist), edges[1:]-edges[:-1], np.ones_like(hist), hist, color=colors)
                                ax.view_init(30,30)
                            else:
                                ax.bar(edges[:-1], hist, edges[1:] - edges[:-1], align='edge')
                        if legend and not dataset.classes.get(tasks[t]) is None:
                            handles = []
                            class_names = {v: k for k, v in dataset.classes[tasks[t]].items()}
                            for i in classes:
                                patch = mpatches.Patch(color=cmap(i), label=class_names[i])
                                handles.append(patch)
                            fig.legend(handles=handles)
                        if not out is None:
                            prefix = out.split('/')[:-1]
                            fig.savefig('/'.join(prefix)+'/dists/'+out.split('/')[-1]+'_%d_%s_dim%d.svg'%(layer,tasks[t], dim))
    #                    plt.close('all')
                        figs.append(fig)
        else:
            if tasks is None:
                dim1, dim2 = get_divs(len(dims))
                fig, axes = plt.subplots(dim1, dim2, figsize=(20,10))
                for i in range(axes.shape[0]):
                    for j in range(axes.shape[1]):
                        current_id = i*dim2 + j
                        hist, edges = np.histogram(zs[:, dims[current_id]], bins=bins)
                        axes[i,j].bar(edges[:-1], hist, edges[1:] - edges[:-1], align='edge')
                        axes[i,j].set_title('axis %d'%dims[current_id])
                if not out is None:
                    prefix = out.split('/')[:-1]
                    fig.savefig(out+'_0.svg'%layer)
                figs.append(fig)
            else:
                dim1, dim2 = get_divs(len(dims))
                for t in range(len(tasks)):
                    class_ids, classes = get_class_ids(dataset, tasks[t], ids=ids)
                    cmap = get_cmap(len(class_ids))
                    if relief:
                        fig, axes = plt.subplots(dim1, dim2, figsize=(20,10), subplot_kw={'projection':'3d'})
                    else:
                        print('hello')
                        fig, axes = plt.subplots(dim1, dim2, figsize=(20,10))
                        
    #                pdb.set_trace()
                    for i in range(axes.shape[0]):
                        dim_y = 0 if len(axes.shape)==1 else axes.shape[1]
                        for j in range(dim_y):
                            current_id = i*dim2 + j
                            for k, cl in enumerate(class_ids):
                                hist, edges = np.histogram(zs[cl, dims[current_id]], bins=bins)
                                colors = cmap(k)
                                if relief:
                                    axes[i,j].bar3d(edges[:-1], k*np.ones_like(hist), np.zeros_like(hist), edges[1:]-edges[:-1], np.ones_like(hist), hist, color=colors, alpha=0.1)
                                    axes[i,j].view_init(30,30)
                                else:
                                    axes[i,j].bar(edges[:-1], hist, edges[1:] - edges[:-1], align='edge')
                                axes[i,j].set_title('axis %d'%dims[current_id])
                            
                    if legend and not dataset.classes.get(tasks[t]) is None:
                        handles = []
                        class_names = {v: k for k, v in dataset.classes[tasks[t]].items()}
                        for i in classes:
                            patch = mpatches.Patch(color=cmap(i), label=class_names[i])
                            handles.append(patch)
                        fig.legend(handles=handles)
    
                    if not out is None:
                        prefix = out.split('/')[:-1]
                        fig.savefig(out+'_%d_%s.svg'%(layer, tasks[t]))
                    figs.append(fig)
    return figs





def plot_class_losses(dataset, model, loss, tasks, loss_task=None, label=None, out=None):
    loss_task = label if loss_task is None else loss_task
    if not issubclass(type(tasks), list):
        tasks = [tasks]
    for task in tasks:
        ids, classes = get_class_ids(dataset, task)
        losses_classwise = {}
        for t in range(len(ids)):
            y = None
            if not label is None:
                y = dataset.metadata.get(label)[ids[t]]
            data = model.format_input_data(dataset.data[ids[t]]);
            y = model.format_label_data(y)
            with torch.no_grad():
                pdb.set_trace()
                vae_out = model.forward(data, y=y)
                if not loss_task is None:
                    loss_y = dataset.metadata.get(loss_task)[ids[t]]
                else:
                    loss_y = None
                pdb.set_trace()
                loss_all, losses = loss.loss(model, vae_out, x=data, y=loss_y)
                for k, v in loss.get_named_losses(losses).items():
                    if not k in losses_classwise.keys():
                        losses_classwise[k] = []
                    losses_classwise[k].append(v)
                    
        n_losses = len(losses_classwise.keys())
        fig = plt.figure('class-wise losses for task %s'%task)
        axes = fig.subplots(n_losses, 1)
        for i, l in enumerate(losses_classwise.keys()):
            axes[i].bar(range(len(ids)), losses_classwise[l], 1)
            plt.xticks(range(len(ids)), classes)
    if not out is None:
        fig.savefig(out+'.svg', format="svg")
    return fig
           


 
def plot_reconstructions(dataset, model, label=None, n_points=10, out=None, preprocessing=None, ids=None):
    n_rows, n_columns = get_divs(n_points)
    ids = np.random.permutation(dataset.data.shape[0])[:n_points] if ids is None else ids
    data = dataset.data[ids]
    metadata = None
    if not label is None:
        metadata = dataset.metadata.get(label)[ids]
    vae_out = model.forward(data, y=metadata)['x_params'][0]
    fig = plt.figure()
    axes = fig.subplots(n_rows, n_columns)
    synth = vae_out[0].cpu().detach().numpy()
    if preprocessing:
        data = preprocessing.invert(data)
        synth = preprocessing.invert(synth)
    for i in range(n_rows):
        for j in range(n_columns):
            axes[i,j].plot(data[i*n_columns+j])
            axes[i,j].plot(synth[i*n_columns+j], linewidth=0.6)
            if len(vae_out) > 1:
                axes[i,j].bar(range(vae_out[0].shape[1]), vae_out[1][i*n_columns+j].cpu().detach().numpy(), align='edge', alpha=0.4)
    if not out is None:
        fig.savefig(out+'.svg', format="svg")
    return fig
            
            
    
    
    
'''
def plot_confusion_matrix(cm, classes,
                          
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def make_confusion_matrix(out, dataset, task):
    cnf_matrix = confusion_matrix(fromOneHot(out['y'].data.numpy()), fromOneHot(dataset.metadata[task]))
    np.set_printoptions(precision=2)
    
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=[str(x) for x in range(10)],
                          title='Confusion matrix, without normalization')
    
'''
