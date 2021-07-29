import argparse
import json
import logging
import sys
import networkx as nx

from pathlib import Path
from nodes.graphnode import build_graph, build_traversal_dfs, build_expanded_traversal
from networkx.drawing.nx_pydot import write_dot

VERSION = '0.5'
logging.basicConfig(format='%(asctime)s, %(levelname)s %(message)s', datefmt='%H:%M:%S')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

LOGGER = logging.getLogger('Pipeline')

def main() -> None:
    parser = init_argparser()
    args = parser.parse_args()
    
    if args.plot_call_graph and not args.graph_output_file:
        raise ValueError('If plotting call graph then the output file must also be specified')
        sys.exit(-1)
    
    call_graph = load_graph_from_file(args.graph_config_path)
    
    result = {}
    
    traversal_order = build_traversal_dfs(call_graph, [], "root")
    LOGGER.debug("Pipeline Traversal Order: %s", traversal_order)
    expanded_traversal_order = build_expanded_traversal(call_graph, [], "root")
    LOGGER.debug("Pipeline Expanded Traversal Order %s", expanded_traversal_order)
    
    if args.plot_call_graph:
        expanded_name = args.graph_output_file + '_expanded.dot'
        g=build_graph_from_list(expanded_traversal_order)
        write_dot(g, expanded_name)
        write_dot(call_graph, args.graph_output_file + '.dot')    
    
    for n_id in traversal_order:
        result = call_graph.nodes[n_id]['data'].execute(result)
     
    return


def build_graph_from_list(input_list):
    graph = nx.DiGraph()
    graph.add_nodes_from(input_list)        
    graph.add_edges_from([(input_list[i], input_list[i+1]) for i in range(len(input_list)-1)])
    return graph


def load_graph_from_file(path_to_graph_config: str) -> nx.DiGraph:
    '''
    Builds the call_graph from the graph configuration file given to the 
    program via command-line

    Returns
    -------
    graph : nx.DiGraph
        The call_graph built from the config file

    '''
       
    try:    
        with open(Path(path_to_graph_config), 'rb') as file_data:
            data = json.load(file_data)
    except FileNotFoundError as err:
        LOGGER.warn('%s', err)
        sys.exit(-1)
    
    return build_graph(data)
            
        
def init_argparser() -> argparse.ArgumentParser:
    """
        Creates an argument parser with all of the possible
        command line arguments that can be passed to AQP

        Returns
        -------
        parser: argparse.ArgumentParser
            Parser to be used to parse arguments
    """

    parser = argparse.ArgumentParser(usage="%(prog)s",description="AQP")
    
    optional = parser.add_argument_group('Optional Arguments')
    optional.add_argument('--graph_config_path', default='config/graph.json')
    optional.add_argument('--plot_call_graph', action='store_true', default=False)
    optional.add_argument('--graph_output_file', default='results/graph')
    optional.add_argument('-v', '--version', action='version', version=f'{parser.prog} version {VERSION}')
    return parser

if __name__ == '__main__':
    main()
