import org.jblas.DoubleMatrix;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;
import java.nio.file.Files;
import java.io.File;

public class NetworkApp {
    static boolean isPrint = true;
    private static final String FILE_DATA_60K = "D:\\Java\\projects\\NN_mnist\\untitled\\src\\train-images.idx3-ubyte";
    private static final String FILE_LABELS_60K = "D:\\Java\\projects\\NN_mnist\\untitled\\src\\train-labels.idx1-ubyte";
    private static final String FILE_DATA_10K = "D:\\Java\\projects\\NN_mnist\\untitled\\src\\t10k-images.idx3-ubyte";
    private static final String FILE_LABELS_10K = "D:\\Java\\projects\\NN_mnist\\untitled\\src\\t10k-labels.idx1-ubyte";

    private static int getRoundDataFromMnist(double st){
        if(st>=0.5)
            return 1;
        else
            return 0;
    }

    private static List<double[][]> getTrainingDataFromMnist() throws IOException {
        List<double[][]> trainingData = new ArrayList<>();


        MnistMatrix[] mnistMatrix = new MnistDataReader().readData(FILE_DATA_60K, FILE_LABELS_60K);
        for (int i = 0; i < mnistMatrix.length; i++) {
            MnistMatrix matrix = mnistMatrix[i];
            double[][] io = new double[2][];
            double[] x = new double[784];
            int[] x_print = new int[784];

            for (int r = 0; r < matrix.getNumberOfRows(); r++) {
                for (int c = 0; c < matrix.getNumberOfColumns(); c++) {
                    x[r * matrix.getNumberOfColumns() + c] = (double) matrix.getValue(r, c) / 255;
                    x_print[r * matrix.getNumberOfColumns() + c] = getRoundDataFromMnist((double) matrix.getValue(r, c) / 255);
                }
            }

            boolean DEBUG = false;

            if(DEBUG){
                for(int j = 0; j < 28*28; j++){
                    System.out.print(x_print[j]);
                    if(j%28==0)
                        System.out.print("\n");
                }
                System.out.print("\n\n\n");
            }

            double[] y = Stream.iterate(0, d -> d).limit(10).mapToDouble(d -> d).toArray();
            y[matrix.getLabel()] = 1;
            io[0] = x;
            io[1] = y;
            trainingData.add(io);
        }


        return trainingData;
    }

    public static DoubleMatrix createArrayfromImage(BufferedImage inImage) {
        double[] arr = new double[inImage.getHeight() * inImage.getWidth()];
        int i = 0;
        for (int y = 0; y < inImage.getHeight(); y++) {
            for (int x = 0; x < inImage.getWidth(); x++) {
                //System.out.print((int)(Math.abs((double) inImage.getRGB(x, y)/16777216)));
                arr[i] = (Math.abs((double) inImage.getRGB(x, y)/16777216));
                //System.out.print(arr[i]);

            }
            //System.out.print("\n");
        }


        DoubleMatrix x = new DoubleMatrix(arr);
        return x;
    }


    public static void main(String[] args) throws IOException {
        List<double[][]> trainingData = getTrainingDataFromMnist();
        SigmoidNetwork net = new SigmoidNetwork(784, 30, 10);
        net.SGD(trainingData, 10, 25, 3.0, trainingData);

        System.out.print("\nХотите начать тест?\n");
        String path = "D:\\neuron_project-master\\train\\000000-num5.png";
        BufferedImage img = ImageIO.read(new File(path));
        System.out.print("\n\n");

        System.out.print(net.test(createArrayfromImage(img)));



    }

}