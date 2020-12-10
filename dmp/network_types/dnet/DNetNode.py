import math
from pprint import pprint

import numpy


class DNetNode:
    
    def __init__(self, num_inputs: int, num_points: int, num_outputs: int, transfer_function):
        print('DNetNode({}*{} -> {})'.format(num_inputs, num_points, num_outputs))
        self.positions: numpy.ndarray = numpy.empty((num_points, num_inputs))
        self.values: numpy.ndarray = numpy.empty((num_outputs, num_points))
        self.transfer_function = transfer_function
    
    def compute(self, input: numpy.ndarray):
        return self.transfer_function(self, input)
        # # get weights
        # result = 0.0
        #
        # # weights = [] * num_points
        # # for point in self.points:
        # #     weights[i] = distance(point.position, input)
        #
        # # positions has each position as a row vector
        #
        # deltas = self.positions - input  # broadcast input
        #
        # # min dimension delta
        # deltas = numpy.abs(deltas)
        # mins = numpy.min()
        #
        # # distances = numpy.sum(deltas ** 2, axis=1)  # row vector of distances
        #
        # # # Voroni method
        # # min_index = numpy.argmin(distances)
        # # value = self.values[:, min_index]
        #
        # # print('layer sizes {} {}'.format(self.positions.shape, self.values.shape))
        # # print('compute:\n{}\n{}\n{}\n{}\n{}\n{}\n{}'.format(input, self.positions, deltas, distances, min_index,
        # #                                                     self.values, value))
        #
        # # multiquadratic method
        # # weights = 1 / (distances + 1)  # or other DNet
        # # weights /= numpy.sum(weights)  # normalize weights
        # # value = numpy.sum(weights * self.values, axis=1)
        # # print('compute\n{}\n{}\n{}\n{}'.format(weights, weights.shape, weights * self.values, value))
        #
        # # quadratic
        # # norm = math.sqrt(self.positions.shape[1] ** 2)
        # # value = 1 / (1 + numpy.sqrt(distances) / norm)  # or other DNet
        # # print('compute\n{}\n{}\n{}'.format(input, distances, value))
        # # weights /= numpy.sum(weights)  # normalize weights
        # # value = numpy.sum(weights * self.values, axis=1)
        # # value = weights
        #
        # # inverse distance weighted using nearest neighbours
        # # n = self.n
        # # nearest = numpy.argparition(distances, n)[:n]
        # #
        # # neighbour_distances = distances[nearest]
        # # neighbour_weights = 1.0 / max_neighbour_distance
        # # neighbour_weights /= numpy.sum(neighbour_weights)
        # # neighbour_values = self.values[nearest]
        # # value = neighbour_values * neighbour_weights
        #
        # # neighbours interpolation
        # # n = self.n
        # # nearest = numpy.argparition(distances, n)[:n]
        # #
        # # neighbour_distances = distances[nearest]
        # # max_neighbour_distance = numpy.max(neighbour_distances)
        # # neighbour_weights = (max_neighbour_distance - neighbour_distances)
        # # neighbour_weights /= max_neighbour_distance
        # # neighbour_values = self.values[nearest]
        # # value = neighbour_values * neighbour_weights
        #
        # return value
    
    @property
    def numParameters(self):
        return self.positions.size + self.values.size
    
    def getFlatParameters(self, parameters):
        numPos = self.positions.size
        parameters[0:numPos] = self.positions.flatten()
        numParameters = numPos + self.values.size
        parameters[numPos:numParameters] = self.values.flatten()
        return numParameters
    
    def setFlatParameters(self, parameters):
        numPos = self.positions.size
        self.positions = parameters[0:numPos].reshape(self.positions.shape)
        numParameters = numPos + self.values.size
        self.values = parameters[numPos:numParameters].reshape(self.values.shape)
        
        print('setFlatParameters', self.positions, self.values)
        return numParameters
    
    @staticmethod
    def scratch(node: 'DNetNode', input: numpy.ndarray):
        deltas = node.positions - input  # broadcast input
        # print('deltas')
        # pprint(deltas)
        weights = numpy.amax(numpy.abs(deltas), axis=1)
        # print('weights')
        # pprint(weights)
        value = numpy.amin(weights)
        # value = numpy.array([numpy.maximum(0.0, numpy.sum(weights * input))])
        # value = numpy.sum(weights * input)
        return value
    
    @staticmethod
    def idw(node: 'DNetNode', input: numpy.ndarray):
        # positions has each position as a row vector
        deltas = node.positions - input  # broadcast input
        distances = numpy.sqrt(numpy.sum(deltas ** 2, axis=1))  # row vector of distances
        weights = 1 / (distances + 1e-100)
        weights /= numpy.sum(weights)
        value = numpy.sum(weights * node.values)
        return value
    
    @staticmethod
    def inverseSquaredDistance(node: 'DNetNode', input: numpy.ndarray):
        # positions has each position as a row vector
        deltas = node.positions - input  # broadcast input
        squaredDistances = numpy.sum(deltas * deltas, axis=1)  # row vector of distances
        weights = 1 / (squaredDistances + 1e-100)
        weights /= numpy.sum(weights)
        value = numpy.sum(weights * node.values)
        # value = numpy.sum(weights)
        return value
    
    @staticmethod
    def rbfTrianglularL1(node: 'DNetNode', input: numpy.ndarray):
        # positions has each position as a row vector
        deltas = node.positions - input  # broadcast input
        
        numPositions = node.positions.shape[0]
        numDimensions = node.positions.shape[1]
        
        avgDistance = (0.5) * numDimensions
        bandwidth = 1.0 / (numPositions * avgDistance)
        
        distances = numpy.sum(numpy.abs(deltas), axis=1)  # row vector of distances
        
        print('positions')
        pprint(node.positions)
        
        # print('input')
        # pprint(input)
        
        # print('deltas')
        # pprint(deltas)
        
        print('distances')
        pprint(distances)
        print('bandwidth {}, numPositions {}, numDimensions {}'.format(bandwidth, numPositions, numDimensions))
        print('avg distance {}, avgDistance {}'.format(numpy.average(distances), avgDistance))
        
        weights = numpy.maximum(1.0 - distances * bandwidth, 0.0)
        
        # weights = 1 / (distances / input.size + 1e-100)
        # weights /= numpy.sum(weights)
        # value = numpy.sum(weights * node.values)
        value = numpy.sum(weights)
        return value
    
    @staticmethod
    def normalizedRbfTrianglularL1(node: 'DNetNode', input: numpy.ndarray):
        # positions has each position as a row vector
        deltas = node.positions - input  # broadcast input
        
        numPositions = node.positions.shape[0]
        numDimensions = node.positions.shape[1]
        
        avgDistance = (0.5) * numDimensions
        bandwidth = 1.0 / (numPositions * avgDistance)
        
        distances = numpy.sum(numpy.abs(deltas), axis=1)  # row vector of distances
        
        # print('positions')
        # pprint(node.positions)
        
        # print('input')
        # pprint(input)
        
        # print('deltas')
        # pprint(deltas)
        
        # print('distances')
        # pprint(distances)
        # print('bandwidth {}, numPositions {}, numDimensions {}'.format(bandwidth, numPositions, numDimensions))
        # print('avg distance {}, avgDistance {}'.format(numpy.average(distances), avgDistance))
        
        weights = numpy.maximum(1.0 - distances * bandwidth, 0.0)
        
        # weights = 1 / (distances / input.size + 1e-100)
        weights /= numpy.sum(weights)
        value = numpy.sum(weights * node.values)
        # value = numpy.sum(weights)
        return value
    
    @staticmethod
    def relu(node: 'DNetNode', input: numpy.ndarray):
        # # positions has each position as a row vector
        # deltas = node.positions - input  # broadcast input
        # squaredDistances = numpy.sum(deltas * deltas, axis=1)  # row vector of distances
        # weights = 1 / (squaredDistances + 1e-100)
        # weights /= numpy.sum(weights)
        # value = numpy.sum(weights * node.values)
        # # value = numpy.sum(weights)
        weights = node.positions[0] / numpy.linalg.norm(node.positions[0])
        value = numpy.array([numpy.maximum(0.0, numpy.sum(weights * input))])
        return value
    
    @staticmethod
    def linear(node: 'DNetNode', input: numpy.ndarray):
        ## positions has each position as a row vector
        # deltas = node.positions - input  # broadcast input
        # squaredDistances = numpy.sum(deltas * deltas, axis=1)  # row vector of distances
        # weights = 1 / (squaredDistances + 1e-100)
        # weights /= numpy.sum(weights)
        # value = numpy.sum(weights * node.values)
        # value /= numpy.sqrt(numpy.sum(weights ** 2))
        # value = numpy.sum(weights)
        weights = node.positions[0].copy()
        # print('weights {}'.format(weights.shape))
        # pprint(weights)
        assert (weights.size == input.size)
        weights /= math.sqrt(weights.size)
        # print('nweights')
        # pprint(weights)
        value = numpy.sum(weights * input)
        # print('value')
        # pprint(value)
        return value
    
    @staticmethod
    def inverseL1(node: 'DNetNode', input: numpy.ndarray):
        # positions has each position as a row vector
        deltas = node.positions - input  # broadcast input
        distances = numpy.sum(numpy.abs(deltas), axis=1)  # row vector of distances
        weights = 1.0 / (distances / deltas.size + 1e-100)
        # weights /= numpy.sum(weights)
        value = numpy.sum(weights * node.values)
        return value
    
    @staticmethod
    def inverseSquaredDistanceBinary(node: 'DNetNode', input: numpy.ndarray):
        # positions has each position as a row vector
        deltas = node.positions - input  # broadcast input
        squaredDistances = numpy.sum(deltas * deltas, axis=1)  # row vector of distances
        weights = 1 / (squaredDistances + 1e-100)
        weights /= numpy.sum(weights)
        values = node.values > .5
        value = numpy.sum(weights * values)
        return value
    
    @staticmethod
    def nearestNeighbour(node: 'DNetNode', input: numpy.ndarray):
        deltas = node.positions - input  # broadcast input
        squaredDistances = numpy.sum(deltas * deltas, axis=1)  # row vector of distances
        min_index = numpy.argmin(squaredDistances)
        value = node.values[:, min_index]
        return value
    
    @staticmethod
    def unnormalizedInverseSquaredDistance(node: 'DNetNode', input: numpy.ndarray):
        # positions has each position as a row vector
        deltas = node.positions - input  # broadcast input
        squaredDistances = numpy.sum(deltas * deltas, axis=1)  # row vector of distances
        weights = 1 / (squaredDistances + 1e-100)
        # weights /= numpy.sum(weights)
        value = numpy.sum(weights * node.values)
        return value
    
    @staticmethod
    def inverseQuadraticDistanceWeighting(node: 'DNetNode', input: numpy.ndarray, epsilon=1.0):
        # positions has each position as a row vector
        deltas = node.positions - input  # broadcast input
        squaredDistances = numpy.sum(deltas ** 2, axis=1)  # row vector of distances
        weights = 1 / (1 + epsilon * squaredDistances)
        weights /= numpy.sum(weights)
        value = numpy.sum(weights * node.values)
        return value
