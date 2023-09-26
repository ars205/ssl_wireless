# Combine 3 datasets for NLOS, LOS, and DIS. Then use the label to cluster the data using t-SNE.
# The data is from the book chapter.
# For system.
import sys
sys.path.append('./')
import numpy as np
import os
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
import importlib
from urllib.request import urlopen  
import argparse
import json
from pathlib import Path
from matplotlib.colors import ListedColormap

# Others in the mean time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, DataLoader

# For torch.
import torch
import torch.nn as nn
from torch.nn import functional as F

import models.model_transformer as wits
importlib.reload(wits)
from models import logger
importlib.reload(logger)
from models.ChannelTransformationModule import ChannelTransformationModule as channel_transforms

# @title [RUN] Validate splits and locations with Plotting

import matplotlib.pyplot as plt
from matplotlib import container
import pandas as pd
import seaborn as sns
sns.reset_orig()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


plt.rcParams['text.usetex'] = True
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "CMM12"})

width_in_inches = 140 / 25.4
height_in_inches = 95 / 25.4

# colors = [
#     (0.8, 0.2, 0.2),   
#     (0.2, 0.8, 0.2),   
#     (0.2, 0.2, 0.8),   
#     (0.8, 0.8, 0.2),   
#     (0.8, 0.2, 0.8),   
#     (0.2, 0.8, 0.8),   
#     (0.8, 0.5, 0.2),   
#     (0.6, 0.6, 0.6)
# ]

colors = [
    (0.4, 0.6, 0.8),  # Light blue
    (0.1, 0.1, 0.6),  # Dark blue
    (0.2, 0.5, 0.7),  # Blue-green
    (0.4, 0.2, 0.7),  # Blue-purple
    (0.9, 0.5, 0.5),  # Light red
    (0.7, 0.1, 0.1),  # Dark red
    (0.9, 0.4, 0.2),  # Red-orange
    (0.9, 0.2, 0.6)   # Red-pink
]



def prep_data_load(args):
  """
  Selected datasets for creating train, test and val sets.
  
  Parameters:
      args:
          - dataset_to_download (str): Dataset to 'download' (DIS_lab_LoS, ULA_lab_LoS, URA_lab_LoS, URA_lab_nLoS).
          - saved_dataset_path (str): Path to where datasets are saved.
          - sub_dataset_file_csi (str): CSI file.
          - sub_dataset_file_loc (str): Locations file.
  
  Returns:
          - train_dataset (data (not) loader torch): Training.
          - val_dataset (data (not) loader torch): Validation.
          - test_dataset (data (not) loader torch): Testing.
  """
  # Define dataset related paths and file names
  dataset_to_download = args.dataset_to_download
  if dataset_to_download == "DIS_lab_LoS":
      download_dataset_sub_path = 'ultra_dense/DIS_lab_LoS'
      channel_file_name = 'ultra_dense/DIS_lab_LoS/samples/channel_measurement_'
      test_set_saved_path = args.saved_dataset_path + "/TestSets/DIS_lab_LoS"
  elif dataset_to_download == "ULA_lab_LoS":
      download_dataset_sub_path = 'ultra_dense/ULA_lab_LoS'
      channel_file_name = 'ultra_dense/ULA_lab_LoS/samples/channel_measurement_'
      test_set_saved_path = args.saved_dataset_path + "/TestSets/ULA_lab_LoS"
  elif dataset_to_download == "URA_lab_LoS":
      download_dataset_sub_path = 'ultra_dense/URA_lab_LoS'
      channel_file_name = 'ultra_dense/URA_lab_LoS/samples/channel_measurement_'
      test_set_saved_path = args.saved_dataset_path + "/TestSets/URA_lab_LoS"
  elif dataset_to_download == "URA_lab_nLoS":
      download_dataset_sub_path = 'ultra_dense/URA_lab_nLoS'
      channel_file_name = 'ultra_dense/URA_lab_nLoS/samples/channel_measurement_'
      test_set_saved_path = args.saved_dataset_path + "/TestSets/URA_lab_nLoS"
  elif dataset_to_download == "S-200":
      print('Note that for this case we use a smaller sample size.')
  elif dataset_to_download == "HB-200":
      print('Note that for this case we use a smaller sample size.')        
  else:
      raise ValueError("This dataset is not used. Check the configuration of dataset name!")

  print(f'Dataset main path is {os.path.dirname(os.path.realpath(args.saved_dataset_path))}')
  print(f'\n\n******** Dataset Selected is {dataset_to_download}************\n\n')

  '''
  Here, you load the data (or a sample from the dataset). Otherwise, below (commented)
  I have put the pre-processing of the KUL datasets.
  '''
  with open(Path(args.saved_dataset_path)/args.sub_dataset_file_csi, 'rb') as f1:
    csi2 = np.load(f1)
    f1.close()
  with open(Path(args.saved_dataset_path)/args.sub_dataset_file_loc, 'rb') as f2:
      location_data_and_classes = np.load(f2)    
      f2.close()


  scalar = MinMaxScaler()
  scalar = scalar.fit(location_data_and_classes[:,0:2])

  #tx_transform = scalar.fit_transform(np.expand_dims(tx_for_normalization[:,0], axis =1))
  tx_transform = scalar.fit_transform(location_data_and_classes[:,0:2])
  # Concat location IDs after scaling
  tx_transform = np.concatenate((tx_transform,location_data_and_classes[:,2:3]), axis=1) 
  # lets create different datasets, if datasets has labels from 0 to 3, set value to 0, if datasets have labels from 4 to 7, set value to 1, otherwise, set value to 2.
    # This is for the KUL datasets.
#   for i in range(len(tx_transform)):
#     if tx_transform[i][2] == 0 or tx_transform[i][2] == 1 or tx_transform[i][2] == 2 or tx_transform[i][2] == 3:
#         tx_transform[i][2] = 0
#     elif tx_transform[i][2] == 4 or tx_transform[i][2] == 5 or tx_transform[i][2] == 6 or tx_transform[i][2] == 7:
#         tx_transform[i][2] = 1
#     else:
#         tx_transform[i][2] = 2
    
  print(csi2.shape, tx_transform.shape)

  # Remove DIS dataset. Becomes too clutered. Otherwise, include spots 8-11.
  mask = ~np.isin(tx_transform[:, 2], [8, 9, 10,11])
  csi2 = csi2[mask]
  tx_transform = tx_transform[mask]


  X_train, x_test, Y_train, y_test = train_test_split(csi2[:,:,0:100,:], tx_transform, stratify=tx_transform[:,2:3], test_size=100) #locations_ID2 was replaced by tx...

  
  #test_size = 15000 used for ablations, otherwise 0.005:
  x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, stratify=Y_train[:,2:3], test_size=100) #1k: 22k; 5k: 18k; 10k: 22k
  print(f"Shapes: {x_train.shape, y_train.shape, x_val.shape, y_val.shape, x_test.shape, y_test.shape}")
  print(f"Train Data: {len(x_train)}")
  print(f"Validation Data: {len(x_val)}")
  print(f"Test Data: {len(x_test)}")
  print("Unique spots for classification: ",np.unique(tx_transform[:,2:3]))



  #train_and_val_batchsize = args.train_val_batchsize #@param {type:"integer"}

  X_train2 = np.einsum('basc->bcas', x_train)
  X_test2 = np.einsum('basc->bcas', x_test)
  X_val2 = np.einsum('basc->bcas', x_val)


  tensor_x = torch.tensor(X_train2).float()
  tensor_test_x = torch.tensor(X_test2).float()
  tensor_val_x = torch.tensor(X_val2).float()

  tensor_y = torch.tensor(y_train).float()#.cuda()
  tensor_test_y = torch.tensor(y_test).float()#.cuda()
  tensor_val_y = torch.tensor(y_val).float()#.cuda()

  train_dataset = TensorDataset(tensor_x,tensor_y) 

  test_dataset = TensorDataset(tensor_test_x,tensor_test_y)# create your datset

  val_dataset = TensorDataset(tensor_val_x,tensor_val_y) # create your datset

  print('done')
  return train_dataset, test_dataset, val_dataset, scalar


# To reset weights, lets initialize initially with xavier, latter we call defaul pytorch rand-normal or pre-trained.
def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)     

### Get embeddings
def get_embeddings(model, h_channel, which_o='LID'):
    """
    Get a compressed representation of the channel (i.e., an embedding).
    
    Parameters
    ----------
    model : object
        Pre-trained model (or random).
    h_channel : tensor
        H tensor (3, N_r, N_c).
    which_o : str
        Options can be to get only the 'LID' and 'mean'. 
          However, you can select any, does not really matter for SWiT.
    
    Returns
    -------
    tensor
        Embedding (1, D).
    """
    # Get the intermediate layer output.
    o_r = model.get_intermediate_layers(h_channel.unsqueeze(0).cuda(), n=1)[0]
    dim = o_r.shape[-1]
    o_r = o_r.reshape(-1, dim)
    
    # Either 'LID' or 'mean'. However, you can select "any" of the "subcarrier" representations. All should give more or less similar results.
    if which_o == 'LID':
        o_r = o_r[0:1, ...]
    else:
        o_r = torch.mean(o_r, 0, True)
        
    return o_r


def get_args_values(args):
    args_dict = vars(args)
    args_strings = []
    for key, value in args_dict.items():
        args_strings.append("{}: {}".format(key, str(value)))
    return "\n".join(args_strings)


if __name__ == '__main__':
  # Load the the config file
  with open("config_tsne.json", "r") as f:
      config = json.load(f)

  results_path = config["project_path"]+"/results/"+config["experiment_name"]
  if not os.path.exists(results_path):
    print("There is no results path for this experiment. Thus, I will create a folder to store the results.\n")
    os.makedirs(results_path)
    print("results path created: ", results_path)

  sys.path.append(config["project_path"])

  CHECKPOINT_PATH = config["project_path"]+"/saved_models/"+config["experiment_name"]  
                  # Path to the folder where the pretrained models are saved
  if not os.path.exists(CHECKPOINT_PATH):
    print("There is no checkpoint path for this experiment. Thus, I will create a folder to store the checkpoints.\n")
    os.makedirs(CHECKPOINT_PATH)
    print("CHECKPOINT_PATH created: ", CHECKPOINT_PATH)
  parser = argparse.ArgumentParser(f'Evaluation on {config["dataset_to_download"]} dataset.')
  parser.add_argument('--experiment_name', type=str, default=config['experiment_name'], help='Name of this experiment.')
  parser.add_argument('--dataset_to_download', type=str, default=config['dataset_to_download'], help='Path to dataset to load.')
  parser.add_argument('--saved_dataset_path', type=str, default=config['saved_dataset_path'], help='Path to dataset to load.')
  parser.add_argument('--sub_dataset_file_csi',type=str, default=config['sub_dataset_to_use'], help='If you already have a subdataset. Avoiding large files.')
  parser.add_argument('--sub_dataset_file_loc',type=str, default=config['sub_loc_dataset_to_use'], help='If you already have a subdataset. Avoiding large files.')
  parser.add_argument('--realMax', type=float, default=config['realMax'], help='Max value of real part for the whole dataset')
  parser.add_argument('--imagMax', type=float, default=config['imagMax'], help='Max value of imag part for the whole dataset')
  parser.add_argument('--absMax', type=float, default=config['absMax'], help='Max value of abs part for the whole dataset')
  parser.add_argument('--model_name', type=str, default=config['arg2_model_name'], help='WiT-based transformer.')
  parser.add_argument("--encoder", type=str, default=config['arg1_encoder'], help='We use momentum target encoder.')
  parser.add_argument('--number_antennas', type=int, default=config['arg3_Nr'], help='Number of antenna elements per subcarrier.')
  parser.add_argument('--total_subcarriers', type=int, default=config['arg4_Nc'], help='Total number of subcarriers.')
  parser.add_argument('--eval_subcarriers', type=int, default=config['arg5_Nc_prime'], help='Selected number of subcarriers.')
  parser.add_argument('--weights_path', type=str, default=CHECKPOINT_PATH+'/checkpoint.pth', help="Path to load pre-trained weights.")
  parser.add_argument('--train_val_batchsize', type=int, default=config['train_and_val_batchsize'], help='Batch size for train and val. For test, we use 1.')
  parser.add_argument("--knn", type=int, default=config['arg_classifier_k'], help="k NNs, default 20.")
  parser.add_argument("--c_spots", type=int, default=config['arg_classifier_c_spots'], help="Number of spots/classes. For KUL: 4; For S: 360; For HB: 406")
  parser.add_argument("--pth_names_classifier", type = list, default = config['pth_names_classifier'], help= "A list of saved models (or epoch checkpoints) to evaluate. Include some random names, to understand the gain.")
  #parser.add_argument("--h_slice",type=tuple, default=(64,1), help="Top kNNs")
  args = parser.parse_args(args=[])

  args.h_slice = (64,1)

  print(f'***Configuration****\n',"\n",get_args_values(args))

  print('\n\n',args.saved_dataset_path)

  train_dataset, test_dataset, val_dataset, scalar = prep_data_load(args)

  # Get Features from pre-trained model.
  datasets = [train_dataset, test_dataset]

  global_transfo2_test = channel_transforms('classifier',realMax=args.realMax, imagMax=args.imagMax, absMax=args.absMax)
  global_transfo2_test_LOS = channel_transforms('classifier',realMax=1.96484375, imagMax=1.9453125, absMax=2.11407)
  global_transfo2_test_DIS = channel_transforms('classifier',realMax=2.015625, imagMax=2.1484375, absMax=2.26824000)


  # To save hyperparameter:
  if len(global_transfo2_test.transform) == 1:
    Transf = str(global_transfo2_test.transform[0])
  elif len(global_transfo2_test.transform) == 2:  
    Transf = str(global_transfo2_test.transform[0]), str(global_transfo2_test.transform[1])
  elif len(global_transfo2_test.transform) == 3:  
    Transf = str(global_transfo2_test.transform[0]), str(global_transfo2_test.transform[1]) ,str(global_transfo2_test.transform[2])  

  knn_results = {'Transformations':[],'Weights':[],'Top1': [],'Top5':[]}

  # Eval. over different models.
  epochs_to_eval = args.pth_names_classifier #['checkpoint_Rnd1','checkpoint0010','checkpoint_Rnd2','checkpoint0030','Rand3','checkpoint0040', 'Rand4','checkpoint0080']#,'checkpoint_nlos,checkpoint_nlos_new']#,'rand','checkpoint0070','checkpoint0090','rrr','checkpoint']

  model = wits.__dict__[args.model_name](h_slice=(args.number_antennas, 1), num_classes=2)

  for j in range(len(epochs_to_eval)):
    
    ### 1. Reset Weights to xavier (to avoid weight accum.) ####
    model.apply(weights_init) 

    ### 2. Load pre-trained weights ###
    args.weights_pth = Path(CHECKPOINT_PATH) / (epochs_to_eval[j]+'.pth') 

    print(f"Model {args.model_name} {args.number_antennas}x{1} built.")
    #model.get_intermediate_layers
    print(f"\nWeights that are supposed to find at : \n {args.weights_pth}\n")
    logger.load_pretrained_weights(model, args.weights_pth, args.encoder)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    # Avoid to get embeddings from a validation set. Only for train and test.
    for i in range(2):
      data_loader = DataLoader(
          datasets[i],
          batch_size=1,
          num_workers=1,
          pin_memory=True,
          drop_last=True,
      )    
      print(f"Data loaded: there are {len(train_dataset)} and {len(test_dataset)} CSI Samples.")

      x_lst = []
      embeddings = []
      spot_lst = []
      for tensor_channel, spot in data_loader:
        h_input = tensor_channel[0:1]
        # If label 0,1,2,or3, then use global_transfo2_test, otherwise, use global_transfo2_test2
        if spot[0][2] == 0 or spot[0][2] == 1 or spot[0][2] == 2 or spot[0][2] == 3:
           h_input = global_transfo2_test(h_input)
        elif spot[0][2] == 4 or spot[0][2] == 5 or spot[0][2] == 6 or spot[0][2] == 7:
           h_input = global_transfo2_test_LOS(h_input)
        else:
              h_input = global_transfo2_test_DIS(h_input)
        h_input = torch.squeeze(h_input, 0)
        h_input_feat = get_embeddings(model.to('cuda'), h_input.to('cuda'), which_o='LID').T
        embeddings.append(h_input_feat.flatten().unsqueeze(0).detach().cpu().numpy())
        x_lst.append(tensor_channel.detach().cpu().numpy())
        spot_lst.append(spot.detach().cpu().numpy())
      print(h_input_feat.shape, h_input.shape)

      if i==0:      
        print("x_lst_num",len(x_lst))
        train_embeddings = np.concatenate(embeddings, axis=0 )
        train_spots = np.concatenate(spot_lst, axis=0 )
    

    print(train_embeddings.shape, train_spots.shape)
    
    top_5 = 100 # Hard coded, since for KUL c_spots = 4.
    print('\n ***Be careful when using other datasets! Uncomment parts where top-5 is set to the value of 100.')
    k=20

    #train_embeddings, train_spots = torch.from_numpy(train_embeddings), torch.from_numpy(train_spots)


    # 4. Create t-SNE Map
    tsne = TSNE(n_components=2, verbose=1, init = 'pca', random_state=123,perplexity=10) 
    data_2D = tsne.fit_transform(train_embeddings)

    # Define your class names
    spot_names = [
        "NLOS $c=1$", "NLOS $c=2$", "NLOS $c=3$", "NLOS $c=4$", 
        "LOS $c=5$", "LOS $c=6$", "LOS $c=7$", "LOS $c=8$"
    ] 

    # Use a perceptually uniform colormap
    #cmap = plt.get_cmap('viridis', len(spot_names))
    cmap = ListedColormap(colors)

    # t-SNE MAP
    #plt.figure(figsize=(10, 8))
    plt.figure(figsize=(width_in_inches, height_in_inches))

    scatter = plt.scatter(data_2D[:, 0], data_2D[:, 1], c=train_spots[:,2], cmap=cmap, alpha=0.6, edgecolors='w', linewidths=0.1)
    plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
    plt.grid(True, which="both", ls="--", alpha=0.3)
    ax = plt.gca()
    ax.axis('off')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    from matplotlib.lines import Line2D
    #legend_elements = [Line2D([0], [0], marker='o', color='w', label=spot_names[i], 
     #                         markersize=10, markerfacecolor=cmap(i)) for i in range(len(spot_names))]
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=spot_names[i], 
                          markersize=10, markerfacecolor=colors[i]) for i in range(len(spot_names))]
    
    #ax.legend(handles=legend_elements, loc=1, prop={'size': 13}, numpoints=1, ncol=2, fancybox=False)
    #ax.legend(handles=legend_elements, loc=1, prop={'size': 11}, ncol=1, fancybox=False)
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), prop={'size': 11}, ncol=1, fancybox=False)
 
    plt.tight_layout()
    # Optionally save the plot
    plt.savefig(results_path+"/tSNE_features_nlos_ura_swit_Rand2.png")
    plt.savefig(results_path+"/tSNE_features_nlos_ura_swit_Rand2.pdf",dpi=400) 
    plt.show() 
    plt.close()
          


