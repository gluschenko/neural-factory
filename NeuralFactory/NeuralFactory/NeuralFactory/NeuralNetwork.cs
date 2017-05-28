using System;
using System.Linq;
using System.Collections.Generic;

namespace NeuralFactory
{
    public class NeuralNetwork
    {
        public double LearnRate { get; set; }
        public double Momentum { get; set; }
        public List<Neuron> InputLayer { get; set; }
        public List<List<Neuron>> HiddenLayers { get; set; }
        public List<Neuron> OutputLayer { get; set; }

        //
        public int StatisticsEpochs = 0;
        public double StatisticsError = 0;
        public double StatisticsTargetError = 0;

        private static readonly Random Random = new Random();

        // Constructor
        public NeuralNetwork(int inputSize, int[] hiddenSizes, int outputSize, double? learnRate = null, double? momentum = null)
        {
            LearnRate = learnRate ?? 0.4;
            Momentum = momentum ?? 0.9;
            InputLayer = new List<Neuron>();
            HiddenLayers = new List<List<Neuron>>();
            OutputLayer = new List<Neuron>();
            
            for (int i = 0; i < inputSize; i++)
            {
                InputLayer.Add(new Neuron(this));
            }

            for (int i = 0; i < hiddenSizes.Length; i++)
            {
                List<Neuron> Layer = new List<Neuron>();

                for (int n = 0; n < hiddenSizes[i]; n++) 
                {
                    List<Neuron> LinkedLayer = InputLayer;
                    if(i > 0)LinkedLayer = HiddenLayers.Last();

                    Layer.Add(new Neuron(this, LinkedLayer));
                }

                HiddenLayers.Add(Layer);
            }

            for (int i = 0; i < outputSize; i++)
            {
                OutputLayer.Add(new Neuron(this, HiddenLayers.Last()));
            }

        }

        // Training

        public void Train(List<DataSet> dataSets, int numEpochs)
        {
            for (int i = 0; i < numEpochs; i++)
            {
                foreach (DataSet dataSet in dataSets)
                {
                    ForwardPropagate(dataSet.Values);
                    BackPropagate(dataSet.Targets);
                }
                //
                StatisticsEpochs++;
            }
        }

        public void Train(List<DataSet> dataSets, double minimumError)
        {
            StatisticsTargetError = minimumError;
            //
            var error = 1.0;
            var numEpochs = 0;

            while (error > minimumError && numEpochs < int.MaxValue)
            {
                var errors = new List<double>();
                foreach (var dataSet in dataSets)
                {
                    ForwardPropagate(dataSet.Values);
                    BackPropagate(dataSet.Targets);
                    errors.Add(CalculateError(dataSet.Targets));
                }
                error = errors.Average();
                numEpochs++;
                //
                StatisticsEpochs++;
                StatisticsError = error;
            }
        }

        private void ForwardPropagate(params double[] inputs)
        {
            int i = 0;
            InputLayer.ForEach(a => a.Value = inputs[i++]);
            foreach(List<Neuron> Layer in HiddenLayers)
            {
                Layer.ForEach(a => a.CalculateValue());
            }

            OutputLayer.ForEach(a => a.CalculateValue());
        }

        private void BackPropagate(params double[] targets)
        {
            int i = 0;
            OutputLayer.ForEach(a => a.CalculateGradient(targets[i++]));
            foreach (List<Neuron> Layer in HiddenLayers) 
            {
                Layer.ForEach(a => a.CalculateGradient());
                Layer.ForEach(a => a.UpdateWeights(LearnRate, Momentum));
            }

            OutputLayer.ForEach(a => a.UpdateWeights(LearnRate, Momentum));
        }

        public double[] Compute(params double[] inputs)
        {
            ForwardPropagate(inputs);
            return OutputLayer.Select(a => a.Value).ToArray();
        }

        private double CalculateError(params double[] targets)
        {
            var i = 0;
            return OutputLayer.Sum(a => Math.Abs(a.CalculateError(targets[i++])));
        }

        // Random
        public static double GetRandom()
        {
            return 2 * Random.NextDouble() - 1;
        }
    }

    public enum TrainingType
    {
        Epoch,
        MinimumError
    }
}