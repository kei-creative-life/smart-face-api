export class Test {
  x: number;
  y: number;
  constructor(x: number, y: number) {
    this.x = x;
    this.y = y;
  }

  add(z: number) {
    return new Test(this.x + z, this.y + z);
  }
}
