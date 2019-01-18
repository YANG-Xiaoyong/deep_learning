package org.deeplearning.layer;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

/**
 * 隐藏层
 * @author yangxy
 *
 */
public class LinearLayer extends AbstractHiddenLayer {
	
	public LinearLayer(Long width) {
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
	public LinearLayer setWbRandom(long cols) {
		if(this.w == null) {
			this.w = Matrix.Factory.randn(this.width, cols);
			this.b = Matrix.Factory.randn(this.width, 1);
		}
		return this;
	}
	/*public LinearLayer setWbRandom(long rows, long cols) {
		if(this.w == null) {
			this.w = Matrix.Factory.ones(rows, cols);
			this.w.randn(Ret.ORIG);
			this.b = Matrix.Factory.ones(rows, 1);
			this.b.randn(Ret.ORIG);
		}
		return this;
	}*/
}
