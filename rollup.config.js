import node from "rollup-plugin-node-resolve";
import typescript from "rollup-plugin-typescript2";
import commonjs from "rollup-plugin-commonjs";
import resolve from "rollup-plugin-node-resolve";

const circularDepFilter = /Circular dependency/;

export default {
  input: "src/index.ts",
  plugins: [
    typescript(),
    node(),
    // Polyfill require() from dependencies.
    commonjs({
      include: 'node_modules/**',
      namedExports: {
        './node_modules/seedrandom/index.js': ['alea']
      },
    })
  ],
  output: {
    extend: true,
    banner: `// @tensorflow/tfjs-layers Copyright ${(new Date).getFullYear()} Google`,
    file: "dist/tf-layers.js",
    format: "umd",
    name: "tf",
    globals: {'crypto': 'crypto', '@tensorflow/tfjs-core': 'tf'}
  },
  external: ['crypto', '@tensorflow/tfjs-core'],
  onwarn: warning => {
    if (circularDepFilter.test(warning.toString())) {
      return;
    }
    console.warn('WARNING: ', warning.toString());
  }
};
