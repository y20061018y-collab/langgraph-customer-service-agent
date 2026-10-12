// 演示页 Markdown 渲染的测试。运行（项目根目录，需要 Node 18+）：
//     node web/markdown.test.js
// 从 index.html 里取出 <markdown-render> 标记之间的代码执行，不需要浏览器或第三方库。
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const assert = require('assert');

const html = fs.readFileSync(path.join(__dirname, 'index.html'), 'utf-8');
const m = /\/\/ <markdown-render>([\s\S]*?)\/\/ <\/markdown-render>/.exec(html);
assert(m, '找不到 <markdown-render> 标记');
const ctx = {};
vm.createContext(ctx);
vm.runInContext(m[1] + '\nthis.renderMarkdown = renderMarkdown;', ctx);
const r = (s) => ctx.renderMarkdown(s);

const ALLOWED = new Set(['p', 'br', 'strong', 'code', 'ul', 'ol', 'li']);
function assertOnlyAllowedTags(out) {
  for (const t of out.matchAll(/<\/?([a-zA-Z][a-zA-Z0-9]*)/g)) {
    assert(ALLOWED.has(t[1].toLowerCase()), `出现了不允许的标签 <${t[1]}>：${out}`);
  }
}

let n = 0;
function test(name, fn) { fn(); n++; console.log('✅ ' + name); }

test('<script> 被转义，不产生标签', () => {
  const out = r('<script>alert(1)</script>');
  assert.strictEqual(out, '<p>&lt;script&gt;alert(1)&lt;/script&gt;</p>');
  assertOnlyAllowedTags(out);
});

test('<b>x</b> 原样显示为文本，不当作 HTML', () => {
  const out = r('<b>x</b>');
  assert.strictEqual(out, '<p>&lt;b&gt;x&lt;/b&gt;</p>');
  assertOnlyAllowedTags(out);
});

test('属性型注入 <img onerror> 被转义', () => {
  const out = r('<img src=x onerror="alert(1)">');
  assert(!out.includes('<img'), out);
  assertOnlyAllowedTags(out);
});

test('未闭合的 ** 原样保留，不产生 strong', () => {
  assert.strictEqual(r('这是 **没有闭合'), '<p>这是 **没有闭合</p>');
  assert.strictEqual(r('** 后面是空格 **'), '<p>** 后面是空格 **</p>');
  assert.strictEqual(r('只有一对 **a** 和一个 ** 落单'), '<p>只有一对 <strong>a</strong> 和一个 ** 落单</p>');
});

test('加粗：含中文引号的成对 **', () => {
  assert.strictEqual(r('系统返回：**"订单 31415 不存在"**'), '<p>系统返回：<strong>&quot;订单 31415 不存在&quot;</strong></p>');
});

test('行内代码，且代码内的 ** 与 < 不被处理', () => {
  assert.strictEqual(r('用 `query_order` 查询'), '<p>用 <code>query_order</code> 查询</p>');
  assert.strictEqual(r('`**a** <b>`'), '<p><code>**a** &lt;b&gt;</code></p>');
});

test('无序列表与有序列表', () => {
  assert.strictEqual(r('- 甲\n- **乙**'), '<ul><li>甲</li><li><strong>乙</strong></li></ul>');
  assert.strictEqual(r('1. 一\n2. 二'), '<ol><li>一</li><li>二</li></ol>');
  assert.strictEqual(r('说明：\n- 甲\n\n结尾'), '<p>说明：</p><ul><li>甲</li></ul><p>结尾</p>');
});

test('换行：单个换行变 <br>，空行分段', () => {
  assert.strictEqual(r('第一行\n第二行'), '<p>第一行<br>第二行</p>');
  assert.strictEqual(r('段一\n\n段二'), '<p>段一</p><p>段二</p>');
  assert.strictEqual(r('a\r\nb'), '<p>a<br>b</p>');
});

test('伪造占位符的控制字符被去掉', () => {
  const out = r('\u00010\u0002 `x`');
  assert(!out.includes('\u0001') && !out.includes('\u0002'), out);
  assertOnlyAllowedTags(out);
});

test('空输入不报错', () => {
  assert.strictEqual(r(''), '');
});

test('综合：真实回复片段只产生白名单标签', () => {
  const out = r('已为您查询订单 **9527**：\n\n- **物流状态**：异常 <i>x</i>\n- 详情：`滞留`\n\n1. 建单\n2. **未闭合');
  assertOnlyAllowedTags(out);
});

console.log(`\n${n} 个测试通过`);
