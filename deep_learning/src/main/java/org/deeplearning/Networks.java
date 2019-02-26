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
import org.ujmp.elasticsearch.ElasticsearchDataMap;


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
	
	public void startTrain(List<Matrix> inputs) {
		while(true) {
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
				this.activeFun.calculateResult(output);//激活函数
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
					Matrix y = nowHiddenLayer.getY();
					Matrix b = nowHiddenLayer.getB();
					Matrix lastLayerDerivative = null;//上一层的导数
					if(i == hiddenLayersSize - 1) {
						lastLayerDerivative = mseLostFun.getDerivative();
					} else {
						lastLayerDerivative = this.hiddenLayers.get(i + 1).getDerivative();
					}

					//计算激活函数的导数
					Matrix reluDerivative =  Matrix.Factory.zeros(y.getRowCount(), y.getColumnCount());
					this.activeFun.calculateDerivative(reluDerivative, y);
					/*for(int rows = 0; rows < y.getRowCount(); rows++) {
						for(int cols = 0; cols < y.getColumnCount(); cols++) {
							if(y.getAsDouble(rows, cols) > 0) {
								reluDerivative.setAsDouble(1, rows, cols);
							} else {
								reluDerivative.setAsDouble(0, rows, cols);
							}
						}
					}*/

					//算入激活函数的导数
					for(int cols = 0; cols < lastLayerDerivative.getColumnCount(); cols++) {
						lastLayerDerivative.setAsDouble(lastLayerDerivative.getAsDouble(0,cols) * reluDerivative.getAsDouble(cols, 0), 0, cols);
					}


					//开始w的梯度下降
					double ϵ = 0.1;
					for(int rows = 0; rows < w.getRowCount(); rows++) {
						for(int cols = 0; cols < w.getColumnCount(); cols++) {
							double fallValue = x.getAsDouble(cols,0) * lastLayerDerivative.getAsDouble(0,rows);//x * 上一层的导数
							if(fallValue > 0) {
								w.setAsDouble(w.getAsDouble(rows,cols) - ϵ * fallValue , rows, cols);
							} else {
								w.setAsDouble(w.getAsDouble(rows,cols) + ϵ * fallValue , rows, cols);
							}
							//dyw.setAsDouble(x.getAsDouble(cols,0) * lastLayerDerivative.getAsDouble(0,rows), rows, cols);
						}
					}

					//开始b的梯度下降
					for(int rows = 0; rows < b.getRowCount(); rows++) {
						double fallValue = lastLayerDerivative.getAsDouble(0,rows);//1 * 上一层的导数
						if(fallValue > 0) {
							b.setAsDouble(b.getAsDouble(rows,0) - ϵ * fallValue , rows, 0);
						} else {
							b.setAsDouble(b.getAsDouble(rows,0) + ϵ * fallValue , rows, 0);
						}
					}

					//设置y对x的导数
					nowHiddenLayer.setDerivative(lastLayerDerivative.mtimes(w));
				}
				/*double ϵ = 0.1;
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
				}*/
				System.out.println("跑了" + this.runCount++);
			} else {
				System.out.println("AA");
				break;
			}
		}
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
