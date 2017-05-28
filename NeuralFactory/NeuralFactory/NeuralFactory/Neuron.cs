using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralFactory
{
    public class Neuron
    {
        public List<Synapse> InputSynapses { get; set; }
        public List<Synapse> OutputSynapses { get; set; }
        public double Bias { get; set; }
        public double BiasDelta { get; set; }
        public double Gradient { get; set; }
        public double Value { get; set; }

        public NeuralNetwork Network { get; set; }

        private Neuron()
        {
            InputSynapses = new List<Synapse>();
            OutputSynapses = new List<Synapse>();
            Bias = NeuralNetwork.GetRandom();
        }

        public Neuron(NeuralNetwork NN)
            : this()
        {
            Network = NN;
        }

        public Neuron(NeuralNetwork NN, IEnumerable<Neuron> inputNeurons) 
            : this()
        {
            Network = NN;

            foreach (var inputNeuron in inputNeurons)
            {
                var synapse = new Synapse(inputNeuron, this);
                inputNeuron.OutputSynapses.Add(synapse);
                InputSynapses.Add(synapse);
            }
        }

        public virtual double CalculateValue()
        {
            return Value = Sigmoid.Output(InputSynapses.Sum(a => a.Weight * a.InputNeuron.Value) + Bias);
        }

        public double CalculateError(double target)
        {
            return target - Value;
        }

        public double CalculateGradient(double? target = null)
        {
            if (target == null)
            {
                //return Gradient = OutputSynapses.Sum(a => a.OutputNeuron.Gradient * a.Weight) * Sigmoid.Derivative(Value);

                List<Synapse> EndOutputSynapses = Network.HiddenLayers.Last()[0].OutputSynapses;
                return Gradient = EndOutputSynapses.Sum(a => a.OutputNeuron.Gradient * a.Weight) * Sigmoid.Derivative(Value);
            }
            
            return Gradient = CalculateError(target.Value) * Sigmoid.Derivative(Value);
        }

        public void UpdateWeights(double learnRate, double momentum)
        {
            var prevDelta = BiasDelta;
            BiasDelta = learnRate * Gradient;
            Bias += BiasDelta + momentum * prevDelta;

            foreach (var synapse in InputSynapses)
            {
                prevDelta = synapse.WeightDelta;
                synapse.WeightDelta = learnRate * Gradient * synapse.InputNeuron.Value;
                synapse.Weight += synapse.WeightDelta + momentum * prevDelta;
            }
        }

    }
}

/*public int ID = 0; //Идентификатор (в пределах одной сети)
       public double Signal = 0; //Место хранения сигнала (трассировка)

       public List<Dendrite> Dendrites = new List<Dendrite>(); //Дендритов много
       public Axon Axon = new Axon(); //Аксон только один
       public NeuronType Type = NeuronType.Linear;
       public Func<double, double> Operation = (s) => { return s; };

       public Neuron()
       {
           //Dendrites = new List<Dendrite>();
       }

       public override string ToString()
       {
           return "[id: " + ID + ", type: " + Type.ToString() + ", dendrites: " + Dendrites.Count + ", axon_dendrites: " + Axon.SynapticDendrites.Count + ", signal: " + Signal + "]";
       }*/
