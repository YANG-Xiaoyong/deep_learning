package org.deeplearning.layer;

import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

/**
 * 输出层
 * @author yangxy
 *
 */
public class DeferentLayer extends AbstractHiddenLayer {
	
	
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
	public DeferentLayer setWbRandom(long rows, long cols) {
		if(this.w == null) {
			this.w = Matrix.Factory.randn(cols, rows);
			this.b = Matrix.Factory.randn(cols, 1);
		}
		return this;
	}
	
}
