package org.deeplearning.layer;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

/**
 * 隐藏层
 * @author yangxy
 *
 */
public class LinearLayer extends AbstractHiddenLayer {
	
	public LinearLayer() {
	}
	
	public LinearLayer(Integer width) {
		super(width);
	}
	
	@Override
	public Matrix calculate() {
		// TODO Auto-generated method stub
		return this.w.mtimes(this.x).plus(this.b);
	}
	
	/**
	 * 随机生成矩阵w和b
	 * @param rows
	 * @param cols
	 * @return
	 */
	public LinearLayer setWbRandom(Integer rows, Integer cols) {
		if(this.w == null) {
			if (this.width == null) {//输出层
				this.width = cols;
			}
			this.w = Matrix.Factory.randn(this.width, rows);
			this.b = Matrix.Factory.randn(this.width, 1);
		}
		return this;
	}
	/*public LinearLayer setWbRandom(long cols) {
		if(this.w == null) {
			this.w = Matrix.Factory.randn(this.width, cols);
			this.b = Matrix.Factory.randn(this.width, 1);
		}
		return this;
	}*/
}
