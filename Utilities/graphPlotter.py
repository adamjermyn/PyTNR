import networkx


def makePlotter(fname, counter=0, positions=None):
    '''
    A graph plotter is a function which plots a graph, saves it to a specific location, and
    maintains a list of node positions to ensure continuity between different versions of a graph.

    This is done by having each graph plotter store this data and return a new graph plotter.
    '''

    def plotter(graph):
        pos = None

        # Matplotlib loads here so that it only loads if the plotter is called.
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        labels = networkx.get_edge_attributes(graph, 'weights')
        for l in labels.keys():
            labels[l] = round(labels[l], 0)
            reusePos = {}

        if positions is not None:
            for n in g.nodes:
                if n in positions:
                    reusePos[n] = pos[n]

            pos = networkx.fruchterman_reingold_layout(
                graph, pos=reusePos, fixed=reusePos.keys())
        else:
            pos = networkx.fruchterman_reingold_layout(graph)

        plt.figure(figsize=(11, 11))
        weights = [graph.edge[i][j]['weight'] **
                   2 / 5 for (i, j) in graph.edges_iter()]
        networkx.draw(graph, pos, width=weights, edge_color=[
                      cm.jet(w / max(weights)) for w in weights])
        networkx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
        plt.savefig(fname + str(counter) + '.pdf')
        plt.close()

        return makePlotter(fname, counter + 1, positions=pos)

    return plotter
