module.exports = {
  testEnvironment: "jsdom",
  setupFilesAfterEnv: ["<rootDir>/src/tests/setup.js"],
  transform: {
    "^.+\\.(js|jsx)$": "babel-jest"
  },
  transformIgnorePatterns: [
    "/node_modules/(?!uuid)/"   // ← ÄLÄ IGNOREOI uuid pakettia
  ],
  moduleFileExtensions: ["js", "jsx"]
};
