// dom_stub.mjs — the smallest fake DOM that lets controls.js run under
// Node's test runner without pulling in a real browser (jsdom, etc).
//
// The frontend is deliberately dependency-free (vanilla JS, no build step —
// see README "Stack"), so its tests should not be the thing that introduces
// the first npm dependency. This stub implements exactly the DOM surface
// controls.js touches: getElementById, a class list, value/textContent, and
// addEventListener/dispatch. Nothing more.

class FakeClassList {
  constructor() { this._set = new Set(); }
  add(c) { this._set.add(c); }
  remove(c) { this._set.delete(c); }
  toggle(c, on) { on ? this._set.add(c) : this._set.delete(c); }
  contains(c) { return this._set.has(c); }
}

class FakeElement {
  constructor(tag = 'div') {
    this.tagName = tag;
    this.classList = new FakeClassList();
    this._listeners = {};
    this._value = '';
    this.textContent = '';
    this.innerHTML = '';
    this.min = undefined;
    this.max = undefined;
    this.dataset = {};
  }
  get value() { return this._value; }
  set value(v) { this._value = String(v); }
  addEventListener(type, fn) {
    (this._listeners[type] ||= []).push(fn);
  }
  dispatch(type) {
    for (const fn of this._listeners[type] || []) fn({ target: this });
  }
  querySelectorAll() { return []; }
}

export function makeControlsDom() {
  const engineSelect = new FakeElement('select');
  const cycleSlider = new FakeElement('input');
  const cycleLabel = new FakeElement('span');
  const playBtn = new FakeElement('button');
  const byId = {
    engineSelect, cycleSlider, cycleLabel, playBtn,
  };
  global.document = {
    getElementById: (id) => byId[id] || null,
    querySelectorAll: () => [],
  };
  return byId;
}
