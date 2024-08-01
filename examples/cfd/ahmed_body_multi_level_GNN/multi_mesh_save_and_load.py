from bsms_graph_wrapper import BistrideMultiLayerGraph
import torch
import os
import pickle
import shutil
from tqdm import tqdm


def cal_multi_mesh_all(graphData,savedir,split,num_layer):
    """
    Precompute the multi-mesh graphs for each input mesh, and save as .pkl.
    """
    dir_name = savedir+"/"+ str(num_layer) + "/"+split
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(f"No precomputed multi-mesh for '{split}', computing and saving now!")

        for (each_graph,each_graph_id) in tqdm(graphData):
            mmfile = os.path.join(dir_name, str(each_graph_id) + '_mmesh_layer_' + str(num_layer) + '.pkl')
            flatten_edges = torch.cat((each_graph.edges()[0].view(1,-1),each_graph.edges()[1].view(1,-1)),dim=0).numpy() #[2,num_edges]
            n=each_graph.num_nodes()
            pos_mesh = each_graph.ndata["pos"].numpy()

            # Initialize BistrideMultiLayerGraph
            multi_layer_graph = BistrideMultiLayerGraph(flatten_edges, num_layer, n, pos_mesh)
            # Get multi-layer graphs
            m_gs, m_flat_es, m_ids = multi_layer_graph.get_multi_layer_graphs()
            # Save
            m_mesh = {'m_gs': m_gs, 'm_flat_es': m_flat_es, 'm_ids': m_ids,}
            pickle.dump(m_mesh, open(mmfile, 'wb'))

    elif os.path.exists(dir_name) and len(graphData)== len([file for file in os.listdir(dir_name) if file.endswith(".pkl")]):
        print(f"Multi-mesh for '{split}' already computed.")
    else:
        print(f"Recomputing multi-mesh for '{split}', due to data mismatch!")
        shutil.rmtree(dir_name)
        os.makedirs(dir_name)

        for (each_graph,each_graph_id) in tqdm(graphData):
            mmfile = os.path.join(dir_name, str(each_graph_id) + '_mmesh_layer_' + str(num_layer) + '.pkl')
            flatten_edges = torch.cat((each_graph.edges()[0].view(1,-1),each_graph.edges()[1].view(1,-1)),dim=0).numpy() #[2,num_edges]
            n=each_graph.num_nodes()
            pos_mesh = each_graph.ndata["pos"].numpy()

            # Initialize BistrideMultiLayerGraph
            multi_layer_graph = BistrideMultiLayerGraph(flatten_edges, num_layer, n, pos_mesh)
            # Get multi-layer graphs
            m_gs, m_flat_es, m_ids = multi_layer_graph.get_multi_layer_graphs()
            # Save
            m_mesh = {'m_gs': m_gs, 'm_flat_es': m_flat_es, 'm_ids': m_ids,}
            pickle.dump(m_mesh, open(mmfile, 'wb'))


   
def load_multi_mesh_batch(graph_id_list,savedir,split,num_layer):
    # loading
    dir_name = savedir+"/"+ str(num_layer) + "/"+split
    mesh_list = []
    # handling batch_size = 1
    if not isinstance(graph_id_list, list):
        graph_id_list = [graph_id_list]
    for graph_id in graph_id_list:
        graph_id = int(graph_id)
        mmfile = os.path.join(dir_name, str(graph_id) + '_mmesh_layer_' + str(num_layer) + '.pkl')
        m_mesh = pickle.load(open(mmfile, 'rb'))
        mesh_list.append(m_mesh)
        #m_gs, m_flat_es, m_ids = m_mesh['m_gs'], m_mesh['m_flat_es'], m_mesh['m_ids']
    return mesh_list
    # TODO: batching

