# -*- coding: utf-8 -*-

import os

import tensorflow as tf
from keras import backend
from keras.models import load_model


def convert_ckpt_to_pb(ckpt_dir, output_dir, output_name, output_nodes):
    '''
    Args:
        ckpt_dir: checkpoint's folder
        output_name: name for output pb file
        output_dir: folder of output pb file
        output_node_names: list of output nodes to be exported
    '''

    if not tf.gfile.Exists(ckpt_dir):
        raise AssertionError(
            'Checkpoint folder not found'
            'Path: {0}'.format(ckpt_dir))

    # Get checkpoint's path
    checkpoint = tf.train.get_checkpoint_state(ckpt_dir)
    input_checkpoint = checkpoint.model_checkpoint_path

    # Check if output folder already exists
    os.makedirs(output_dir, exist_ok=True) 
    output_graph = os.path.join(output_dir, output_name)

    # Create a new session with empty graph
    with tf.Session(graph=tf.Graph()) as sess:
        # Load model from checkpoint
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)

        # Restore parameters from checkpoint
        saver.restore(sess, input_checkpoint)
        graph_def = tf.get_default_graph().as_graph_def()

        # Check if output nodes are specified
        if not output_nodes:
            raise AttributeError('Please specify a list of output nodes')
        else:
            output_nodes = output_nodes.split(',')

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            graph_def,
            output_nodes # filter graph by output nodes
        )

        # Write pb file to disk
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())

def convert_h5_to_pb(h5_path, output_dir, output_name):
    '''
    Args:
        h5_path: path to h5 file
        output_name: name for output pb file
        output_dir: folder of output pb file
    '''

    model = load_model(h5_path)
    frozen_graph = freeze_session(backend.get_session(), output_names=[out.op.name for out in model.outputs])
    tf.io.write_graph(frozen_graph, output_dir, output_name, as_text=False)

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    '''
    This method is copied from the following question on Stackoverflow:
    https://stackoverflow.com/a/45466355/4510614

    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                        or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    '''

    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph
