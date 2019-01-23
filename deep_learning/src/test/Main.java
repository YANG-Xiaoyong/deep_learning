import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.deeplearning.Networks;
import org.deeplearning.activeFun.ReLUActiveFun;
import org.deeplearning.layer.AbstractHiddenLayer;
import org.deeplearning.layer.DeferentLayer;
import org.deeplearning.layer.LinearLayer;
import org.deeplearning.lossfun.MseLostFun;
import org.ujmp.core.Matrix;

public class Main {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		List<Matrix> x = new ArrayList<Matrix>();
		for(int i = 0; i < 1000; i++) {
			Matrix t = Matrix.Factory.randn(2, 1);//训练数据，把x、y都放在一个维度里
			x.add(t);
		}
	    Matrix y = Matrix.Factory.zeros(x.size(), 1);//测试集
	    for (int i = 0; i < x.size(); i++) {
	    	y.setAsDouble(testFun2(x.get(i).getAsDouble(0, 0), x.get(i).getAsDouble(1,0)), i, 0);
		}
	    
	    Networks deepNetworks = new Networks();
	    MseLostFun mseLossFun = new MseLostFun();
	    ReLUActiveFun activeFun = new ReLUActiveFun();
	    //设置激活函数
	    deepNetworks.setActiveFun(activeFun);
	    //设置测试集
	    mseLossFun.setExpectedResult(y);
	    //设置神经网络层
	    List<AbstractHiddenLayer> lineLayers = new ArrayList<>();
	    lineLayers.add(new LinearLayer(10));
	    lineLayers.add(new LinearLayer(3));
	    lineLayers.add(new LinearLayer(7));
	    lineLayers.add(new LinearLayer(4));
	    lineLayers.add(new LinearLayer());
	    /*for(int i = 0; i < 4; i++) {//添加隐藏层
	    	lineLayers.add(new LinearLayer(10L));
	    }*/
	    //lineLayers.add(new DeferentLayer());//添加输出层
	    deepNetworks.setHiddenLayers(lineLayers);
	    //设置损失函数
	    deepNetworks.setLostFun(mseLossFun);
	    
	    while(deepNetworks.statrHiddenLayer(x)) {
	    }    
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

}
