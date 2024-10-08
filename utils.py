import os
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures

fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
Feature_family_name = list(factory.GetFeatureDefs().keys())
def smile_to_hgraph(smile):
    mol = Chem.MolFromSmiles(smile)

    AllChem.Compute2DCoords(mol)

    feats = factory.GetFeaturesForMol(mol)
    edges = [[],[]]
    for col in range(len(feats)):
        for row in feats[col].GetAtomIds():
            edges[0].append(row)
            edges[1].append(col)
    return edges

class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/data',dataset=None,
                 xd=None, y=None, transform=None,
                 pre_transform=None,smile_graph=None,
                 text_embedding=None):

        #root is required for save preprocessed data, default is '/tmp'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        self.text_embedding = text_embedding
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, y,smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass
        #return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    # Customize the process method to fit the task of matabolic pathway types prediction
    # Inputs:
    # xd - list of SMILES
    # Y: list of labels
    # Return: PyTorch-Geometric format processed data
    def process(self, xd, y,smile_graph):
        assert (len(xd) == len(y)), "The two lists must be the same length!"
        data_list = []
        data_len = len(xd)
        for i in range(data_len):
            print('Converting SMILES to graph: {}/{}'.format(i+1, data_len))
            smiles = xd[i]
            labels = y[i]
            text_em=self.text_embedding[i]
            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index,smi_em,fp= smile_graph[smiles]
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            h_index=smile_to_hgraph(smiles)
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                hedge_index=torch.LongTensor(h_index),
                                smi_em=torch.Tensor(smi_em),
                                fp=torch.FloatTensor(fp),
                                y=torch.FloatTensor([labels]),
                                text_em=torch.Tensor(text_em))
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            # append graph, label and target sequence to data list
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])

