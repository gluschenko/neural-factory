using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Drawing;
using System.Windows.Forms;
using Threading = System.Threading;
using NeuralFactory;
using Anvil;

namespace NeuralFactory
{
    public class Program
    {
        public static Form MainForm;
        public static int FormWidth = 800, FormHeight = 600;

        public static string Version = "0.1";
        public static string FormTitle = string.Format("Neural Factory (v {0})", Version);

        public static PictureBox GUIPictureBox;

        public static Timer UpdateTimer = new Timer();

        [STAThread]
        public static void Main()
        {
            AnvilCore.Init();
            //
            MainForm = UI.CreateForm(new Rect(0, 0, FormWidth, FormHeight), FormTitle);
            MainForm.Size = new Size(FormWidth, FormHeight);
            MainForm.MinimumSize = new Size(100, 100);

            GUIPictureBox = UI.CreatePictureBox(new Rect(0, 0, MainForm.ClientSize.Width, MainForm.ClientSize.Height), UI.Anchor(true, true, true, true), "GUI");

            UI.Append(GUIPictureBox, MainForm);
            //
            App.Start();
            //
            UpdateTimer.Interval = 60;
            UpdateTimer.Tick += delegate
            {
                GUI.Update(GUIPictureBox);
                App.Update();
                GC.Collect();
            };

            UpdateTimer.Start();
            //
            Application.Run(MainForm);
        }

        public class App
        {
            static Threading.Thread TrainingThread;

            static double DrawingScale = 1;
            static bool canDrawNetwork = true;
             
            static NeuralNetwork NN;
            static List<DataSet> Data;

            static Timer ExecutionTimer = new Timer();
            static string ExecutionInfo = "";

            static void InitNN() 
            {
                NN = new NeuralNetwork(5, new int[] { 5, 3 }, 1, 0.2, 0);

                Data = new List<DataSet> {

                     new DataSet(new double[]{
                         0, 0, 0, 0, 0
                     },
                     new double[]{
                         0
                     }),

                     new DataSet(new double[]{
                         0, 0, 0, 0, 1
                     },
                     new double[]{
                         0.1
                     }),

                     new DataSet(new double[]{
                         0, 0, 0, 1, 0
                     },
                     new double[]{
                         0.2
                     }),

                     new DataSet(new double[]{
                         0, 0, 0, 1, 1
                     },
                     new double[]{
                         0.3
                     }),

                     new DataSet(new double[]{
                         0, 0, 1, 0, 0
                     },
                     new double[]{
                         0.4
                     }),

                     new DataSet(new double[]{
                         0, 0, 1, 0, 1
                     },
                     new double[]{
                         0.5
                     }),

                     new DataSet(new double[]{
                         0, 0, 1, 1, 0
                     },
                     new double[]{
                         0.6
                     }),

                     new DataSet(new double[]{
                         0, 0, 1, 1, 1
                     },
                     new double[]{
                         0.7
                     }),

                     new DataSet(new double[]{
                         0, 1, 0, 0, 0
                     },
                     new double[]{
                         0.8
                     }),

                     new DataSet(new double[]{
                         0, 1, 0, 0, 1
                     },
                     new double[]{
                         0.9
                     }),

                     new DataSet(new double[]{
                         0, 1, 0, 1, 0
                     },
                     new double[]{
                         1.0
                     }),
                 };
            }

            //

            static void StopTraining()
            {
                if (TrainingThread != null)
                {
                    TrainingThread.Abort();
                    TrainingThread = null;
                }
            }

            static void StartTraining() 
            {
                if (TrainingThread != null) 
                {
                    TrainingThread.Abort();
                    TrainingThread = null;
                }

                TrainingThread = new Threading.Thread(TrainingProc);
                TrainingThread.IsBackground = true;

                TrainingThread.Start(new KeyValuePair<NeuralNetwork, List<DataSet>>(NN, Data));
            }

            static void TrainingProc(object args) 
            {
                KeyValuePair<NeuralNetwork, List<DataSet>> InArgs = (KeyValuePair<NeuralNetwork, List<DataSet>>)args;
                NeuralNetwork Network = InArgs.Key;
                List<DataSet> Data = InArgs.Value;

                //NN.Train(Data, int.MaxValue);
                NN.Train(Data, 0.01);
            }

            //

            //public static Vector2 ScrollOffset = new Vector2(0, 0);

            public static void Start()
            {
                Program.MainForm.Icon = Properties.Resources.ICON;

                Styles.Init();

                InitNN();

                //

                ExecutionTimer.Interval = 100;
                ExecutionTimer.Tick += delegate {
                    TraceNetwork();
                };
                ExecutionTimer.Start();
            }

            public static void Update()
            {
                GUI.Clear(Color.Black);
                //
                if (canDrawNetwork)
                {
                    DrawNetwork();
                }
                //
                if (TrainingThread == null)
                {
                    if (GUI.Button(new Rect(GUI.Width - 200, GUI.Height - 45, 120, 40), "Start training"))
                    {
                        StartTraining();
                    }
                }
                else
                {
                    if (GUI.Button(new Rect(GUI.Width - 200, GUI.Height - 45, 120, 40), "Stop training"))
                    {
                        StopTraining();
                    }

                }

                if (GUI.Button(new Rect(GUI.Width - 75, GUI.Height - 45, 70, 40), "Reset"))
                {
                    StopTraining();
                    InitNN();
                }

                //

                if (GUI.Button(new Rect(5, GUI.Height - 45, 40, 40), "+"))
                {
                    DrawingScale *= 1.1;
                }

                if (GUI.Button(new Rect(50, GUI.Height - 45, 40, 40), "-"))
                {
                    DrawingScale *= 0.9;
                }

                if (!canDrawNetwork)
                {
                    if (GUI.Button(new Rect(100, GUI.Height - 45, 60, 40), "Show"))
                    {
                        canDrawNetwork = true;
                    }
                }
                else 
                {
                    if (GUI.Button(new Rect(100, GUI.Height - 45, 60, 40), "Hide"))
                    {
                        canDrawNetwork = false;
                    }
                }

                GUI.Label(new Rect(10, 10, 1000, 1000), ExecutionInfo, Styles.FreeLabel);
            }

            public static void TraceNetwork() 
            {
                string Output = "";
                List<double> OutputErrors = new List<double>();

                foreach (DataSet DS in Data)
                {
                    double[] RealOutputs = NN.Compute(DS.Values);
                    for (int i = 0; i < RealOutputs.Length; i++) RealOutputs[i] = Math.Round(RealOutputs[i], 3);

                    Output += string.Join(", ", DS.Values) + " => " + string.Format("{0} ({1})\n", string.Join(", ", DS.Targets), string.Join(", ", RealOutputs));
                    //
                    List<double> Errors = new List<double>();
                    for (int i = 0; i < DS.Targets.Length; i++)
                    {
                        Errors.Add(Math.Abs(DS.Targets[i] - RealOutputs[i]));
                    }
                    OutputErrors.Add(Errors.Average());
                }

                Output += string.Format("\nAccuracy: {0}%", 100 - (OutputErrors.Average() * 100.0));
                Output += string.Format("\nLearn rate: {0}", NN.LearnRate);
                Output += string.Format("\nMomentum: {0}", NN.Momentum);
                Output += string.Format("\nEpochs: {0}", NN.StatisticsEpochs);
                Output += string.Format("\nError: {0}", NN.StatisticsError);
                Output += string.Format("\nTarget error: {0}", NN.StatisticsTargetError);

                ExecutionInfo = Output;
            }

            struct NeuronView
            {
                public bool isNeuron;
                public Point DrawingPoint;
                public double DrawingRatio;
                public string DrawingTip;
                public string DrawingFace;
            }

            public static void DrawNetwork()
            {
                string NeuronTip = "";
                Point TipScreenPosition = new Point();

                //

                List<List<Neuron>> Layers = new List<List<Neuron>>();
                Layers.Add(NN.InputLayer);
                Layers.AddRange(NN.HiddenLayers.ToArray());
                Layers.Add(NN.OutputLayer);

                //
                
                List<List<NeuronView>> DrawingPoints = new List<List<NeuronView>>();
                List<List<List<double>>> DrawingWeights = new List<List<List<double>>>();

                int MaxNeuronsNumber = 0;
                foreach (List<Neuron> L in Layers)
                {
                    if (L.Count > MaxNeuronsNumber) MaxNeuronsNumber = L.Count;
                }

                int HorizontalSpacing = (int)(40 * DrawingScale);
                int VerticalSpacing = (int)(40 * DrawingScale);
                int NeuronSize = (int)(20 * DrawingScale);

                int DrawingWidth = HorizontalSpacing * (Layers.Count - 1);
                int DrawingHeight = VerticalSpacing * (MaxNeuronsNumber - 1);


                for (int l = 0; l < Layers.Count; l++)
                {
                    List<Neuron> Layer = Layers[l];
                    List<NeuronView> Points = new List<NeuronView>();
                    List<List<double>> Weights = new List<List<double>>();

                    int DisplayedNeuronsLimit = 16;

                    for (int n = 0; n < Layer.Count; n++)
                    {
                        if (Layer.Count <= DisplayedNeuronsLimit)
                        {
                            int x = (GUI.Width / 2) - (DrawingWidth / 2) + (l * HorizontalSpacing);
                            int y = (GUI.Height / 2) - (DrawingHeight / 2) + ((VerticalSpacing * (MaxNeuronsNumber - Layer.Count)) / 2) + (n * VerticalSpacing);

                            Points.Add(new NeuronView
                            {
                                isNeuron = true,
                                DrawingPoint = new Point(x, y),
                                DrawingRatio = Layer[n].Value,
                                DrawingTip = string.Format("Value: {0}\nBias: {1}\nGradient: {2}", Layer[n].Value, Layer[n].Bias, Layer[n].Gradient),
                                DrawingFace = "~",
                            });
                        }
                        //
                        List<double> W = new List<double>();
                        foreach (Synapse S in Layer[n].OutputSynapses)
                        {
                            W.Add(S.Weight);
                        }
                        Weights.Add(W);
                    }

                    if (Layer.Count > DisplayedNeuronsLimit)
                    {
                        int x = (GUI.Width / 2) - (DrawingWidth / 2) + (l * HorizontalSpacing);
                        int y = (GUI.Height / 2);

                        Points.Add(new NeuronView
                        {
                            isNeuron = false,
                            DrawingPoint = new Point(x, y),
                            DrawingRatio = 0,
                            DrawingTip = "",
                            DrawingFace = Layer.Count.ToString(),
                        });
                    }

                    DrawingPoints.Add(Points);
                    DrawingWeights.Add(Weights);
                }

                //

                for (int L = 0; L < DrawingPoints.Count; L++)
                {
                    int PIterator = 0;
                    foreach (NeuronView P in DrawingPoints[L])
                    {
                        int NextPIterator = 0;
                        if (L < DrawingPoints.Count - 1)
                        {
                            Point LastPoint = new Point();
                            Point LastNextPoint = new Point();

                            foreach (NeuronView NextP in DrawingPoints[L + 1])
                            {
                                double W = DrawingWeights[L][PIterator][NextPIterator];

                                if (LastPoint != P.DrawingPoint || LastNextPoint != NextP.DrawingPoint)
                                {
                                    double AW = Math.Abs(W);
                                    AW = Math.Sqrt(AW); // Чтобы вес не разрастался на графе
                                    GUI.DrawLine(P.DrawingPoint, NextP.DrawingPoint, (int)AW);
                                }

                                LastPoint = P.DrawingPoint;
                                LastNextPoint = NextP.DrawingPoint;

                                NextPIterator++;
                            }
                        }

                        PIterator++;
                    }
                }

                foreach (List<NeuronView> L in DrawingPoints)
                {
                    foreach (NeuronView P in L)
                    {
                        GUIStyle NStyle = Styles.NeuronView;
                        NStyle.NormalBackgroundColor = Rainbow((1 - P.DrawingRatio) * 0.15);

                        Rect R = new Rect(P.DrawingPoint.X - (NeuronSize / 2), P.DrawingPoint.Y - (NeuronSize / 2), NeuronSize, NeuronSize);

                        GUI.Button(R, P.DrawingFace, NStyle);
                        if(Input.Hover(R))
                        {
                            /*Point CursorPos = Input.GetCursorPosition();
                            GUI.Box(new Rect(CursorPos.X, CursorPos.Y, 200, 40), "");
                            GUI.Label(new Rect(CursorPos.X + 10, CursorPos.Y + 10, 180, 20), P.DrawingRatio.ToString());*/

                            TipScreenPosition = Input.GetCursorPosition();
                            NeuronTip = P.DrawingTip;
                        }
                    }
                }

                //
                if (NeuronTip != "")
                {
                    GUI.Box(new Rect(TipScreenPosition.X, TipScreenPosition.Y, 300, 160), "");
                    GUI.Label(new Rect(TipScreenPosition.X + 10, TipScreenPosition.Y + 10, 280, 140), NeuronTip);
                }
                NeuronTip = "";
            }

            public static Color Rainbow(double progress)
            {
                double div = (Math.Abs(progress % 1) * 6);
                int ascending = (int)((div % 1) * 255);
                int descending = 255 - ascending;

                switch ((int)div)
                {
                    case 0:
                        return Color.FromArgb(255, 255, ascending, 0);
                    case 1:
                        return Color.FromArgb(255, descending, 255, 0);
                    case 2:
                        return Color.FromArgb(255, 0, 255, ascending);
                    case 3:
                        return Color.FromArgb(255, 0, descending, 255);
                    case 4:
                        return Color.FromArgb(255, ascending, 0, 255);
                    default: // case 5:
                        return Color.FromArgb(255, 255, 0, descending);
                }
            }

            //

            public class Styles
            {
                public static GUIStyle NeuronView = new GUIStyle
                {
                    NormalBackgroundColor = GUIColors.Orange,
                    HoverBackgroundColor = GUIColors.Orange,
                    NormalColor = GUIColors.White,
                    BorderRadius = 1000,
                };

                public static GUIStyle BlueNeuronView = new GUIStyle
                {
                    NormalBackgroundColor = GUIColors.Blue,
                    HoverBackgroundColor = GUIColors.Blue,
                    NormalColor = GUIColors.White,
                    BorderRadius = 1000,
                };

                public static GUIStyle FreeLabel = new GUIStyle
                {
                    FontSize = 10,
                    NormalColor = GUIColors.White,
                };

                public static GUIStyle Divider = new GUIStyle
                {
                    NormalBackgroundColor = GUIColors.Gray,
                };

                public static GUIStyle GreenButton = GUISkin.CreateButton();
                public static GUIStyle RedButton = GUISkin.CreateButton();

                public static void Init()
                {
                    GreenButton.NormalBackgroundColor = GUIColors.Green;
                    GreenButton.HoverBackgroundColor = GUIColors.GreenHover;
                    GreenButton.ActiveBackgroundColor = GUIColors.Green;

                    RedButton.NormalBackgroundColor = GUIColors.Red;
                    RedButton.HoverBackgroundColor = GUIColors.RedHover;
                    RedButton.ActiveBackgroundColor = GUIColors.Red;
                }
            }
        }
    }
}