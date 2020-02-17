from copy import deepcopy

from TNR.TensorLoopOptimization.optimizer import optimize as opt
from TNR.Environment.environment import artificialCut, identityEnvironment, fullEnvironment

def loop_svd_optimize_network(network, node, return_copy):
    if return_copy:
        network = deepcopy(network)
        node = list(n for n in network.nodes if n.id == node.id)[0]

    return network, loop_svd_optimize_node(node, False)

def loop_svd_optimize_node(node, return_copy):
    if return_copy:
        node = deepcopy(node)

    node.tensor = loop_svd_optimize(node.tensor, False)

    return node
    
def loop_svd_optimize(tensor, return_copy):
    if return_copy:
        tensor = deepcopy(tensor)

    cycles = sorted(networkx.cycles.cycle_basis(tensor.network.toGraph()), key=len)

    # Really want to go over all small cycles, but unclear how to generate them.
    #cycles = networkx.cycles.simple_cycles(networkx.DiGraph(self.network.toGraph()))

    for loop in cycles:
        if len(affected.intersection(loop)) > 0 and len(loop) < 0:

            print(loop)

            #print('Optimizing cycle of length',len(loop))

            environment, net, internalBids, envBids = identityEnvironment(tensor, loop)

            print(net)

            #environment, net, internalBids, envBids = artificialCut(self, loop)
            bids = list([b.id for b in net.externalBuckets])

            #print(net)

            # Optimize
            #arr = self.array #####
            #arr0 = net.array
            #net0 = deepcopy(net)
            net, inds = opt(net, 1e-12, environment, bids, envBids)
            #bidDict = {b.id:i for i,b in enumerate(net.externalBuckets)}
            #iDict = {i:bid for i,bid in enumerate(bids)}
            #net.externalBuckets = list(net.externalBuckets[bidDict[iDict[i]]] for i in range(len(bids)))
            #arr01 = net.array

            #print('Err_internal',L2error(arr0,arr01))

            # Throw the new tensors back in
            num = 0
            for m in tensor.network.nodes:
                for n in net.network.nodes:
                    if n.id == m.id:
                        m.tensor = n.tensor
                        num += 1

            #environment, net, internalBids, envBids = identityEnvironment(self, loop)
            #bidDict = {b.id:i for i,b in enumerate(net.externalBuckets)}
            #iDict = {i:bid for i,bid in enumerate(bids)}
            #net.externalBuckets = list(net.externalBuckets[bidDict[iDict[i]]] for i in range(len(bids)))                        
            #arr11 = net.array
            #print(arr0 / np.max(np.abs(arr0)))
            #print(arr11 / np.max(np.abs(arr0)))

            #print('Err_internal_set',L2error(arr0,arr11))
            # Either something is going wrong with the above insertion procedure
            # or else the network is somehow hypersensitive to small components.
            # That couuld be captured by the environment but the degree (of order
            # 1e15) is surprising, especially on small networks where Z ~ 5000.
            # Is something going wrong with the above insertion proceduure?

            #arr2 = self.array  #####
            #print('Original (Err):',arr,arr2)
            #environment, net2, internalBids, envBids = fullEnvironment(self, loop)
            #print('Err Angle:',np.exp(self.logNorm - environment.logNorm - net2.logNorm))
            #print('Err Angle:',np.exp(self.logNorm - environment.logNorm - net0.logNorm))
            #print('Err2',L2error(arr,arr2))

            assert num == len(loop)

    return tensor