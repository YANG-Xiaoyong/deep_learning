package org.deeplearning;

import java.util.List;
import java.util.Random;

import org.deeplearning.activeFun.AbstractActiveFun;
import org.deeplearning.activeFun.ReLUActiveFun;
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
public class Networks {
	
	private static Random random = new Random();
	
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
		MseLostFun mseLostFun = ((MseLostFun)this.lostFun);
		for (int i = 0; i < hiddenLayersSize; i++) {
			Matrix input = null;
			Matrix output = null;
			AbstractHiddenLayer nowHiddenLayer = this.hiddenLayers.get(i);
			if(i == 0) {
				int randNum = this.random.nextInt(inputs.size());
				input = inputs.get(randNum);
				Integer[] expectedIndex = new Integer[]{randNum};
				mseLostFun.setExpectedIndex(expectedIndex);
			} else {
				input = this.hiddenLayers.get(i-1).getY();
			}
			nowHiddenLayer.setX(input);
			int rows = (int) input.getRowCount();
			int cols = (int) input.getColumnCount();
			output = ((LinearLayer)nowHiddenLayer).setWbRandom(rows, cols).calculate();
			//((ReLUActiveFun)this.activeFun).invoke(output);//激活函数
			nowHiddenLayer.setY(output);
		}
		mseLostFun.setComputeResult(this.hiddenLayers.get(hiddenLayersSize - 1).getY());
		double mseResult = mseLostFun.invoke();
		System.out.println("MSE result：" + mseResult);
		if(mseResult > mseLostFun.getThreshold()) {
			for(int i = hiddenLayersSize - 1; i >= 0; i--) {
				AbstractHiddenLayer nowHiddenLayer = this.hiddenLayers.get(i);
				Matrix x = nowHiddenLayer.getX();//y = wx + b，对w逐元素求导，得到x值
				Matrix w = nowHiddenLayer.getW();
				Matrix derivative = Matrix.Factory.zeros(1, x.getColumnCount());
				Matrix dyw = Matrix.Factory.zeros(w.getRowCount(),w.getColumnCount());//y对w求导再乘以上一层y对x的求导
				Matrix lastLayerDerivative = null;//上一层的导数
				if(i == hiddenLayersSize - 1) {
					lastLayerDerivative = mseLostFun.getDerivative();
					/*for(int j = 0; j < lastLayerDerivative.getRowCount(); j++) {
						for(int k = 0; k < x.getRowCount(); k++) {
							double partialDerivativeValue = lastLayerDerivative.getAsDouble(j,0) * x.getAsDouble(k,j);//求偏导值
							derivative.setAsDouble(partialDerivativeValue, k, j);
						}
					}*/
				} else {
					lastLayerDerivative = this.hiddenLayers.get(i + 1).getDerivative();
					/*for(int j = 0; j < lastLayerDerivative.getRowCount(); j++) {
						for(int k = 0; k < x.getRowCount(); k++) {
							double partialDerivativeValue = lastLayerDerivative.getAsDouble(j,0) * x.getAsDouble(k,j);//求偏导值
							derivative.setAsDouble(partialDerivativeValue, k, j);
						}
					}*/
				}
				for(int rows = 0; rows < dyw.getRowCount(); rows++) {
					for(int cols = 0; cols < dyw.getColumnCount(); cols++) {
						dyw.setAsDouble(x.getAsDouble(cols,0) * lastLayerDerivative.getAsDouble(0,rows), rows, cols);
					}
				}
				nowHiddenLayer.setDerivative(lastLayerDerivative.mtimes(w));
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
