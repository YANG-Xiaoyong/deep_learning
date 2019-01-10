package entity;

import org.ujmp.core.Matrix;

public abstract class HiddenLayer {

	/**
	 * m by n Matrix：input
	 */
	protected Matrix x;
	
	protected Matrix getX() {
		return x;
	}

	protected void setX(Matrix x) {
		this.x = x;
	}
	
	abstract Matrix calculate();
	
}
