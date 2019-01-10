import java.util.ArrayList;
import java.util.List;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

import entity.DeepNetworks;
import entity.HiddenLayer;
import entity.LineLayer;
import service.impl.MseLossFunImpl;

public class Main {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
	    Matrix x = Matrix.Factory.ones(4, 1);//训练数据
	    x.rand(Ret.ORIG);
	    Matrix y = Matrix.Factory.ones(4, 1);//测试集
	    for (int i = 0; i < x.getRowCount(); i++) {
	    	y.setAsDouble(testFun(x.getAsDouble(i,0)), i, 0);
		}
	    
	    DeepNetworks deepNetworks = new DeepNetworks();
	    MseLossFunImpl mseLossFun = new MseLossFunImpl();
	    //设置测试集
	    mseLossFun.setTestSet(y);
	    //设置神经网络层
	    List<HiddenLayer> lineLayers = new ArrayList<>();
	    lineLayers.add(new LineLayer());
	    lineLayers.add(new LineLayer());
	    lineLayers.add(new LineLayer());
	    lineLayers.add(new LineLayer());
	    deepNetworks.setHiddenLayers(lineLayers);
	    //设置损失函数
	    deepNetworks.setLossFun(mseLossFun);
	    
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

}
