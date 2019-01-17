package org.deeplearning;

import java.util.List;

import org.deeplearning.actveFun.AbstractActiveFun;
import org.deeplearning.actveFun.ReLUActiveFun;
import org.deeplearning.layer.AbstractHiddenLayer;
import org.deeplearning.layer.DeferentLayer;
import org.deeplearning.layer.LinearLayer;
import org.deeplearning.lossfun.AbstractLostFun;
import org.deeplearning.lossfun.MseLostFun;
import org.ujmp.core.Matrix;


/**
 * 神经网络
 * @author upcif003
 *
 */
public class DeepNetworks {
	
	private int runCount = 1;

	/**
	 * 隐藏层
	 */
	private List<AbstractHiddenLayer> hiddenLayers;
	
	
	/**
	 * 损失函数
	 */
	private AbstractLostFun lostFun;
	
	/**
	 * 激活函数
	 */
	private AbstractActiveFun activeFun;
	
	public boolean statrHiddenLayer(List<Matrix> inputs) {
		//System.out.println(input);
		int hiddenLayersSize = this.hiddenLayers.size();
		for (int i = 0; i < hiddenLayersSize; i++) {
			Matrix input = null;
			Matrix output = null;
			AbstractHiddenLayer nowHiddenLayer = this.hiddenLayers.get(i);
			if(i == 0) {
				input = inputs.get(0);
			} else {
				input = this.hiddenLayers.get(i-1).getY();
			}
			nowHiddenLayer.setX(input);
			if(nowHiddenLayer instanceof LinearLayer) {
				long rows = input.getRowCount();
				output = ((LinearLayer)nowHiddenLayer).setWbRandom(rows, rows).calculate();
			} else if (nowHiddenLayer instanceof DeferentLayer) {
				long rows = input.getRowCount();
				long cols = input.getColumnCount();
				output = ((DeferentLayer)nowHiddenLayer).setWbRandom(rows, cols).calculate();
			}
			((ReLUActiveFun)this.activeFun).invoke(output);//激活函数
			nowHiddenLayer.setY(output);
			//System.out.println(output);
		}
		MseLostFun mseLostFun = ((MseLostFun)this.lostFun);
		mseLostFun.setResultSet(this.hiddenLayers.get(hiddenLayersSize - 1).getY());
		double mseResult = mseLostFun.invoke();
		System.out.println("MSE result：" + mseResult);
		if(mseResult > mseLostFun.getThreshold()) {
			for(int i = hiddenLayersSize - 1; i >= 0; i--) {
				AbstractHiddenLayer nowHiddenLayer = this.hiddenLayers.get(i);
				Matrix x = nowHiddenLayer.getX();//y = wx + b，对w逐元素求导，得到x值
				Matrix derivative = Matrix.Factory.ones(x.getRowCount(), x.getColumnCount());
				Matrix lastLayerDerivative = null;//上一层的导数
				if(i == hiddenLayersSize - 1) {
					lastLayerDerivative = mseLostFun.getDerivative();
				} else {
					lastLayerDerivative = ((LinearLayer)this.hiddenLayers.get(i + 1)).getDerivative();
				}
				for(int j = 0; j < x.getRowCount(); j++) {
					for(int k = 0; k < x.getColumnCount(); k++) {
						double partialDerivativeValue = lastLayerDerivative.getAsDouble(j,0) * x.getAsDouble(j,k);//求偏导值
						derivative.setAsDouble(partialDerivativeValue, j, k);
					}
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
			System.out.println("跑了" + this.runCount++);
			statrHiddenLayer(inputs);//继续训练
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

	public AbstractLostFun getLostFun() {
		return lostFun;
	}

	public void setLostFun(AbstractLostFun lostFun) {
		this.lostFun = lostFun;
	}

	public AbstractActiveFun getActiveFun() {
		return activeFun;
	}

	public void setActiveFun(AbstractActiveFun activeFun) {
		this.activeFun = activeFun;
	}
	
}
