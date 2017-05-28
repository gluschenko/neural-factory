using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralFactory
{
    // This class contains data of each layer
    public class Layer
    {
        public int Width = 1;
        public int Height = 1;
        public int Depth = 1;
        public LayerConnectionType ConnectionType = LayerConnectionType.Forward;
        public Neuron[,,] Neurons;

        // Layers with default connection type
        public Layer(int width)
        {
            Width = width;
        }

        public Layer(int width, int height) : this(width)
        {
            Height = height;
        }

        public Layer(int width, int height, int depth) : this(width, height)
        {
            Depth = depth;
        }

        // Layers with optional connection type
        public Layer(int width, LayerConnectionType connection) : this(width)
        {
            ConnectionType = connection;
        }

        public Layer(int width, int height, LayerConnectionType connection) : this(width, height)
        {
            ConnectionType = connection;
        }

        public Layer(int width, int height, int depth, LayerConnectionType connection) : this(width, height, depth)
        {
            ConnectionType = connection;
        }
    }

    // Type of connection with next layer
    public enum LayerConnectionType
    {
        Forward = 0x0,
        Linear = 0x10,
        Convolution = 0x20,
        Empty = 0x100,
    }
}
