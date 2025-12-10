module.exports = function (context, options) {
  return {
    name: 'docusaurus-plugin-api-proxy',
    configureWebpack(config, isServer, utils) {
      return {
        devServer: {
          proxy: [
            {
              context: ['/api'],
              target: 'http://localhost:8000',
              changeOrigin: true,
              secure: false,
            }
          ],
        },
      };
    },
  };
};
