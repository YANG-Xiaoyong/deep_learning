package org.deeplearning;

import java.util.List;

import org.deeplearning.layer.AbstractHiddenLayer;
import org.deeplearning.layer.LinearLayer;
import org.deeplearning.lossfun.LossFun;
import org.deeplearning.lossfun.MseLossFunImpl;
import org.ujmp.core.Matrix;


/**
 * 神经网络
 * @author upcif003
 *
 */
public class DeepNetworks {

	/**
	 * 隐藏层
	 */
	private List<AbstractHiddenLayer> hiddenLayers;
	
	
	/**
	 * 损失函数
	 */
	private LossFun lossFun;
	
	public boolean statrHiddenLayer(Matrix input) {
		//System.out.println(input);
		Matrix output = input;
		int hiddenLayersSize = this.hiddenLayers.size();
		for (int i = 0; i < hiddenLayersSize; i++) {
			AbstractHiddenLayer nowHiddenLayer = this.hiddenLayers.get(i);
			nowHiddenLayer.setX(output);
			long count = output.getRowCount();
			if(nowHiddenLayer instanceof LinearLayer) {
				output = ((LinearLayer)nowHiddenLayer).setWbRandom(count, count).calculate();
			}
			//System.out.println(output);
		}
		MseLossFunImpl mseLossFun = ((MseLossFunImpl)this.lossFun);
		mseLossFun.setResultSet(this.hiddenLayers.get(hiddenLayersSize - 1).getX());
		double mseResult = mseLossFun.invoke();
		System.out.println("MSE result：" + mseResult);
		if(mseResult > mseLossFun.getThreshold()) {
			for(int i = hiddenLayersSize - 1; i >= 0; i--) {
				LinearLayer nowHiddenLayer = (LinearLayer)this.hiddenLayers.get(i);
				Matrix x = nowHiddenLayer.getX();//y = wx + b，对w逐元素求导，得到x值
				Matrix derivative = Matrix.Factory.ones(x.getRowCount(), x.getColumnCount());
				Matrix lastLayerDerivative = null;//上一层的导数
				if(i == hiddenLayersSize - 1) {
					lastLayerDerivative = mseLossFun.getDerivative();
				} else {
					lastLayerDerivative = ((LinearLayer)this.hiddenLayers.get(i + 1)).getDerivative();
				}
				for(int j = 0; j < x.getRowCount(); j++) {
					double partialDerivativeValue = lastLayerDerivative.getAsDouble(j,0) * x.getAsDouble(j,0);//求偏导值
					derivative.setAsDouble(partialDerivativeValue, j, 0);
					//System.out.println("partialDerivativeValue:" + partialDerivativeValue);
				}
				nowHiddenLayer.setDerivative(derivative);
			}
			double ϵ = 0.1;
			for(int i = 0; i < hiddenLayersSize; i++) {//开始梯度下降
				LinearLayer nowHiddenLayer = (LinearLayer)this.hiddenLayers.get(i);
				Matrix w = nowHiddenLayer.getW();
				Matrix derivative = nowHiddenLayer.getDerivative();
				for(int j = 0; j < w.getRowCount(); j++) {
					if(derivative.getAsDouble(j,0) > 0) {
						w.setAsDouble(w.getAsDouble(j,0) - ϵ * derivative.getAsDouble(j,0) , j, 0);
					} else {
						w.setAsDouble(w.getAsDouble(j,0) + ϵ * derivative.getAsDouble(j,0) , j, 0);
					}
					
				}
			}
			statrHiddenLayer(input);//继续训练
			return false;
		} 
		return true;
			
		
	}

	public List<AbstractHiddenLayer> getHiddenLayers() {
		return hiddenLayers;
	}

	public void setHiddenLayers(List<AbstractHiddenLayer> hiddenLayers) {
		this.hiddenLayers = hiddenLayers;
	}

	public LossFun getLossFun() {
		return lossFun;
	}

	public void setLossFun(LossFun lossFun) {
		this.lossFun = lossFun;
	}
	
	
}
