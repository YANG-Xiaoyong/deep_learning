import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.deeplearning.Networks;
import org.deeplearning.activeFun.IdenticalActiveFun;
import org.deeplearning.activeFun.ReLUActiveFun;
import org.deeplearning.layer.AbstractHiddenLayer;
import org.deeplearning.layer.LinearLayer;
import org.deeplearning.lossfun.MseLostFun;
import org.ujmp.core.Matrix;

public class Main {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		List<Matrix> x = new ArrayList<Matrix>();
		for(int i = 0; i < 1000; i++) {
			Matrix t = Matrix.Factory.randn(2, 1);//训练数据
			x.add(t);
		}
	    List<Matrix> y = new ArrayList<>();//测试集结果
	    for (int i = 0; i < x.size(); i++) {
	    	Matrix result = Matrix.Factory.zeros(1, 1);
	    	result.setAsDouble(testFun3(x.get(i).getAsDouble(0, 0), x.get(i).getAsDouble(1,0)), 0, 0);
	    	y.add(result);
		}
		
	    
	    Networks deepNetworks = new Networks();
	    MseLostFun mseLossFun = new MseLostFun();
	    ReLUActiveFun activeFun = new ReLUActiveFun();//激活函数放在线性层
	    IdenticalActiveFun identicalActiveFun = new IdenticalActiveFun();//恒等函数放在线性层
	    //设置激活函数
	    deepNetworks.setActiveFun(identicalActiveFun);
	    //设置测试集
	    mseLossFun.setExpectedResult(y);
	    //设置神经网络层
	    List<AbstractHiddenLayer> lineLayers = new ArrayList<>();
	    //各种异常情况要考虑
	    /* lineLayers.add(new LinearLayer(10));
	    lineLayers.add(new LinearLayer(3));
	    lineLayers.add(new LinearLayer(7));
	    lineLayers.add(new LinearLayer(4));
	    lineLayers.add(new LinearLayer(5));
	    lineLayers.add(new LinearLayer(8));*/
	    Random random = new Random();
	    for (int i = 0; i < 1; i++) {
	    	//lineLayers.add(new LinearLayer(random.nextInt(10) + 1));
		}
	    lineLayers.add(new LinearLayer());
	    deepNetworks.setHiddenLayers(lineLayers);
	    //设置损失函数
	    deepNetworks.setLostFun(mseLossFun);
	    
	    deepNetworks.startTrain(x);
	}
	
	/**
	 * 一元一次方程
	 * @param x
	 * @return
	 */
	public static double testFun(double x) { 
		return 2 * x + 1;
	}
	
	/**
	 * 二元一次方程
	 * @param x
	 * @param y
	 * @return
	 */
	public static double testFun2(double x,double y) { 
		return x*y + 1;
	}
	
	/**
	 * 线性方程
	 * @param x
	 * @param y
	 * @return
	 */
	public static double testFun3(double x,double y) { 
		return 2*x + 3*y;
	}

}
