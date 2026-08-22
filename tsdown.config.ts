import { defineConfig } from 'tsdown'

export default defineConfig({
  entry: ['src/index.ts'],
  format: 'esm',
  // `using` declarations must be lowered: the stock DSH CLI's Node runtime
  // rejects the raw syntax (mount E2E caught this on 0.1.1-rc.2).
  target: 'es2022',
  dts: true,
  clean: true,
  deps: {
    neverBundle: true,
  },
})
