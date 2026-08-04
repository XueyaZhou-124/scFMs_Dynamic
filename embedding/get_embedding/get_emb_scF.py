# edited from scFoundation github

# Copyright 2023 BioMap (Beijing) Intelligence Technology Limited
MODEL_PATH = '/macroverse/public/zhouxy/pretrained_models/scFoundation/model'
import sys
sys.path.insert(0, MODEL_PATH)

import argparse
import random,os
import numpy as np
import pandas as pd
import argparse
import torch
from tqdm import tqdm
import scipy.sparse
from scipy.sparse import issparse
import scanpy as sc
from load import *
import yaml
from ..utils.data_utils import getadata


# usage
# python get_emb_scF.py --task_name Mouse_hematopoiesis --input_type singlecell --output_type cell --pool_type all --tgthighres t4 \
# --data_path /personal/scF_dynamic/data/mouse_hematopoiesis/exp1_invitro_timecourse/neu_mono_onlylineage_scF_input.h5ad \
# --save_path /personal/scF_dynamic/output/mouse_hematopoiesis/ --pre_normalized T --version ce --model_path ~/scFoundation/model/models/models.ckpt

def run(config):
    args = argparse.Namespace()

    task_name = config["task_name"]
    args.task_name = task_name
    model_name = config['model']

    args.input_type = config['embedding'].get("input_type", "singlecell")
    args.output_type = config['embedding'].get("output_type", "cell")
    args.pool_type = config['embedding'].get("pool_type", "all")
    args.tgthighres = config['embedding'].get("tgthighres", "t4")

    args.data_path = config['embedding']['dataset_path']
    args.save_path = config['embedding']['output_path']
    
    args.pre_normalized = config['embedding'].get("pre_normalized", "F")
    args.version = config['embedding'].get("version", "ce")
    args.model_path = config['embedding']["model_path"] # ckpt path
    args.ckpt_name = config.get("ckpt_name", "01B-resolution")

    # select col
    args.select_col = config['embedding'].get("select_col", None)

    print(args)
    main(args)



####################################Settings#################################
def parse_args():
    parser = argparse.ArgumentParser(description='Drug_response_pre')

    parser.add_argument('--task_name', type=str, default='deepcdr', help='task name')
    parser.add_argument('--input_type', type=str, default='singlecell',choices=['singlecell','bulk'], help='input type; default: singlecell')
    parser.add_argument('--output_type', type=str, default='cell',choices=['cell','gene','gene_batch','gene_expression'], help='cell or gene embedding; default: cell the difference between gene and gene_batch is that in gene mode the gene embedding will be processed one by one. while in gene_batch mode, the gene embedding will be processed in batch. GEARS use gene_batch mode.')
    parser.add_argument('--pool_type', type=str, default='all',choices=['all','max'], help='pooling type of cell embedding; default: all only valid for output_type=cell')
    parser.add_argument('--tgthighres', type=str, default='t4', help='the targeted high resolution (start with t) or the fold change of the high resolution (start with f), or the addtion (start with a) of the high resoultion. only valid for input_type=singlecell')
    parser.add_argument('--data_path', type=str, default='./', help='input data path')
    parser.add_argument('--save_path', type=str, default='./', help='save path')
    parser.add_argument('--pre_normalized', type=str, default='F',choices=['F','T','A'], help='if normalized before input; default: False (F). choice: True(T), Append(A) When input_type=bulk: pre_normalized=T means log10(sum of gene expression). pre_normalized=F means sum of gene expression without normalization. When input_type=singlecell: pre_normalized=T or F means gene expression is already normalized+log1p or not. pre_normalized=A means gene expression is normalized and log1p transformed. the total count is appended to the end of the gene expression matrix.')
    parser.add_argument('--version',  type=str, default='ce', help='only valid for output_type=cell. For read depth enhancemnet, version=rde For others, version=ce')
    parser.add_argument('--model_path',  type=str, default='None', help='pre-trained model path')
    parser.add_argument('--ckpt_name',  type=str, default='01B-resolution', help='checkpoint name')

    # select col
    parser.add_argument('--select_col', default=None, help='select meta info')


    args = parser.parse_args()

    return args


def main_gene_selection(X_df, gene_list):
    """
    Describe:
        rebuild the input adata to select target genes encode protein 
    Parameters:
        adata->`~anndata.AnnData` object: adata with var index_name by gene symbol
        gene_list->list: wanted target gene 
    Returns:
        adata_new->`~anndata.AnnData` object
        to_fill_columns->list: zero padding gene
    """
    X_df = X_df.loc[:, pd.notna(X_df.columns)]
    if X_df.columns.has_duplicates:
        X_df = X_df.T.groupby(level=0).mean().T

    to_fill_columns = list(set(gene_list) - set(X_df.columns))
    padding_df = pd.DataFrame(np.zeros((X_df.shape[0], len(to_fill_columns))), 
                              columns=to_fill_columns, 
                              index=X_df.index)
    X_df = pd.DataFrame(np.concatenate([df.values for df in [X_df, padding_df]], axis=1), 
                        index=X_df.index, 
                        columns=list(X_df.columns) + list(padding_df.columns))
    X_df = X_df[gene_list]
    
    var = pd.DataFrame(index=X_df.columns)
    var['mask'] = [1 if i in to_fill_columns else 0 for i in list(var.index)]
    return X_df, to_fill_columns,var


def main(args):
    #Set random seed
    random.seed(0)
    np.random.seed(0)  # numpy random generator

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gene_list_df = pd.read_csv(os.path.join(MODEL_PATH, 'OS_scRNA_gene_index.19264.tsv'), header=0, delimiter='\t')
    gene_list = list(gene_list_df['gene_name'])

    #Load data
    if args.data_path[-3:]=='npz':
        gexpr_feature = scipy.sparse.load_npz(args.data_path)
        gexpr_feature = pd.DataFrame(gexpr_feature.toarray())
    elif args.data_path[-4:]=='h5ad':
        gexpr_feature = sc.read_h5ad(args.data_path)
        # meta info
        meta = gexpr_feature.obs
        if args.select_col is not None:
            select_col = args.select_col
            meta = meta.loc[:,select_col]

        idx = gexpr_feature.obs_names.tolist()
        col = gexpr_feature.var.index.tolist()
        if issparse(gexpr_feature.X):
            gexpr_feature = gexpr_feature.X.toarray()
        else:
            gexpr_feature = gexpr_feature.X
        gexpr_feature = pd.DataFrame(gexpr_feature,index=idx,columns=col)

    elif args.data_path[-3:]=='npy':
        gexpr_feature = np.load(args.data_path)
        gexpr_feature = pd.DataFrame(gexpr_feature)
    else:
        gexpr_feature=pd.read_csv(args.data_path,index_col=0)
    
    if gexpr_feature.shape[1] != 19264:
        print('Convert gene features to fixed 19264-dim input')
        gexpr_feature, to_fill_columns,var = main_gene_selection(gexpr_feature,gene_list)
        assert gexpr_feature.shape[1]>=19264
    
    if (args.pre_normalized == 'F') and (args.input_type == 'bulk'):  # normalize bulk input
        adata = sc.AnnData(gexpr_feature)
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        gexpr_feature = pd.DataFrame(adata.X,index=adata.obs_names,columns=adata.var_names)

    print(gexpr_feature.shape)

    #Load model
    if args.version == 'noversion':
        ckpt_path = args.model_path
        key=None
    else:
        # ckpt_path = './models/models.ckpt'
        ckpt_path = args.model_path
        if args.output_type == 'cell':
            if args.version == 'ce':
                key = 'cell'
            elif args.version == 'rde':
                key = 'rde'
            else:
                raise ValueError('No version found')
        elif args.output_type == 'gene':
            key = 'gene'
        elif args.output_type == 'gene_batch':
            key = 'gene'
        elif args.output_type == 'gene_expression': # Not recommended
            key = 'gene'
        else:
            raise ValueError('output_mode must be one of cell gene, gene_batch, gene_expression')
    pretrainmodel,pretrainconfig = load_model_frommmf(ckpt_path,key)  # load checkpoint and key
    pretrainmodel.eval()

    geneexpemb=[]
    batchcontainer = []

    # edited
    if os.path.isdir(args.save_path):
        strname = os.path.join(args.save_path, args.task_name +'_'+ args.ckpt_name +"_"+ args.input_type + '_' + args.output_type + '_embedding_' + args.tgthighres + '_resolution.h5ad')  # AnnData output
    else:
        strname = args.save_path

    print('save at {}'.format(strname))
    
    #Inference
    for i in tqdm(range(gexpr_feature.shape[0])):
        with torch.no_grad():
            #Bulk
            if args.input_type == 'bulk':
                if args.pre_normalized == 'T':
                    totalcount = gexpr_feature.iloc[i,:].sum()
                elif args.pre_normalized == 'F':
                    totalcount = np.log10(gexpr_feature.iloc[i,:].sum())
                else:
                    raise ValueError('pre_normalized must be T or F')
                tmpdata = (gexpr_feature.iloc[i,:]).tolist()
                pretrain_gene_x = torch.tensor(tmpdata+[totalcount,totalcount]).unsqueeze(0).cuda()
                data_gene_ids = torch.arange(19266, device=pretrain_gene_x.device).repeat(pretrain_gene_x.shape[0], 1)
            
            #Single cell
            elif args.input_type == 'singlecell':
                # pre-Normalization
                if args.pre_normalized == 'F':
                    tmpdata = (np.log1p(gexpr_feature.iloc[i,:]/(gexpr_feature.iloc[i,:].sum())*1e4)).tolist()
                elif args.pre_normalized == 'T':
                    tmpdata = (gexpr_feature.iloc[i,:]).tolist()
                elif args.pre_normalized == 'A':
                    tmpdata = (gexpr_feature.iloc[i,:-1]).tolist()
                else:
                    raise ValueError('pre_normalized must be T,F or A')

                if args.pre_normalized == 'A':
                    totalcount = gexpr_feature.iloc[i,-1]
                else:
                    totalcount = gexpr_feature.iloc[i,:].sum()

                # select resolution
                if args.tgthighres[0] == 'f':
                    pretrain_gene_x = torch.tensor(tmpdata+[np.log10(totalcount*float(args.tgthighres[1:])),np.log10(totalcount)]).unsqueeze(0).cuda()
                elif args.tgthighres[0] == 'a':
                    pretrain_gene_x = torch.tensor(tmpdata+[np.log10(totalcount)+float(args.tgthighres[1:]),np.log10(totalcount)]).unsqueeze(0).cuda()
                elif args.tgthighres[0] == 't':
                    pretrain_gene_x = torch.tensor(tmpdata+[float(args.tgthighres[1:]),np.log10(totalcount)]).unsqueeze(0).cuda()
                else:
                    raise ValueError('tgthighres must be start with f, a or t')
                data_gene_ids = torch.arange(19266, device=pretrain_gene_x.device).repeat(pretrain_gene_x.shape[0], 1)

            value_labels = pretrain_gene_x > 0
            x, x_padding = gatherData(pretrain_gene_x, value_labels, pretrainconfig['pad_token_id'])
            #Cell embedding
            if args.output_type=='cell':
                position_gene_ids, _ = gatherData(data_gene_ids, value_labels, pretrainconfig['pad_token_id'])
                x = pretrainmodel.token_emb(torch.unsqueeze(x, 2).float(), output_weight = 0) 
                position_emb = pretrainmodel.pos_emb(position_gene_ids)
                x += position_emb
                geneemb = pretrainmodel.encoder(x,x_padding)

                geneemb1 = geneemb[:,-1,:]
                geneemb2 = geneemb[:,-2,:]
                geneemb3, _ = torch.max(geneemb[:,:-2,:], dim=1)
                geneemb4 = torch.mean(geneemb[:,:-2,:], dim=1)
                if args.pool_type=='all':
                    geneembmerge = torch.concat([geneemb1,geneemb2,geneemb3,geneemb4],axis=1)
                elif args.pool_type=='max':
                    geneembmerge, _ = torch.max(geneemb, dim=1)
                else:
                    raise ValueError('pool_type must be all or max')
                geneexpemb.append(geneembmerge.detach().cpu().numpy())

            #Gene embedding
            elif args.output_type=='gene':
                pretrainmodel.to_final = None
                encoder_data, encoder_position_gene_ids, encoder_data_padding, encoder_labels, decoder_data, decoder_data_padding, new_data_raw, data_mask_labels, decoder_position_gene_ids = getEncoerDecoderData(pretrain_gene_x.float(),pretrain_gene_x.float(),pretrainconfig)
                out = pretrainmodel.forward(x=encoder_data, padding_label=encoder_data_padding,
                            encoder_position_gene_ids=encoder_position_gene_ids,
                            encoder_labels=encoder_labels,
                            decoder_data=decoder_data,
                            mask_gene_name=False,
                            mask_labels=None,
                            decoder_position_gene_ids=decoder_position_gene_ids,
                            decoder_data_padding_labels=decoder_data_padding,
                            )
                out = out[:,:19264,:].contiguous()
                geneexpemb.append(out.detach().cpu().numpy())

            #Gene batch embedding
            elif args.output_type=='gene_batch':
                batchcontainer.append(pretrain_gene_x.float())
                if len(batchcontainer)==gexpr_feature.shape[0]:
                    batchcontainer = torch.concat(batchcontainer,axis=0)
                else:
                    continue
                pretrainmodel.to_final = None
                encoder_data, encoder_position_gene_ids, encoder_data_padding, encoder_labels, decoder_data, decoder_data_padding, new_data_raw, data_mask_labels, decoder_position_gene_ids = getEncoerDecoderData(batchcontainer,batchcontainer,pretrainconfig)
                out = pretrainmodel.forward(x=encoder_data, padding_label=encoder_data_padding,
                            encoder_position_gene_ids=encoder_position_gene_ids,
                            encoder_labels=encoder_labels,
                            decoder_data=decoder_data,
                            mask_gene_name=False,
                            mask_labels=None,
                            decoder_position_gene_ids=decoder_position_gene_ids,
                            decoder_data_padding_labels=decoder_data_padding,
                            )
                geneexpemb = out[:,:19264,:].contiguous().detach().cpu().numpy()
            #Gene_expression
            elif args.output_type=='gene_expression':
                encoder_data, encoder_position_gene_ids, encoder_data_padding, encoder_labels, decoder_data, decoder_data_padding, new_data_raw, data_mask_labels, decoder_position_gene_ids = getEncoerDecoderData(pretrain_gene_x.float(),pretrain_gene_x.float(),pretrainconfig)
                out = pretrainmodel.forward(x=encoder_data, padding_label=encoder_data_padding,
                            encoder_position_gene_ids=encoder_position_gene_ids,
                            encoder_labels=encoder_labels,
                            decoder_data=decoder_data,
                            mask_gene_name=False,
                            mask_labels=None,
                            decoder_position_gene_ids=decoder_position_gene_ids,
                            decoder_data_padding_labels=decoder_data_padding,
                            )
                out = out[:,:19264].contiguous()
                geneexpemb.append(out.detach().cpu().numpy())                
            else:
                raise ValueError('output_type must be cell or gene or gene_batch or gene_expression')
    geneexpemb = np.squeeze(np.array(geneexpemb))
    hidden_dim = geneexpemb.shape[1]
    print(geneexpemb.shape)

    # edited
    if args.output_type=='cell':
        print('Concatenating meta columns')
        geneexpemb = pd.DataFrame(geneexpemb, index = idx)
        geneexpemb = pd.concat([geneexpemb, meta],axis=1)
        adata = getadata(geneexpemb, hidden_dim, meta.columns.to_list())
        sc.write(strname, adata)
        print(f'save embedding to {strname}')

    else:
        strname = strname.replace('.h5ad', '.npy')
        np.save(strname,geneexpemb)


if __name__=='__main__':
    args = parse_args()
    main()
