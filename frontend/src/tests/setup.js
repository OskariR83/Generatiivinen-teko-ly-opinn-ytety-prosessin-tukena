import '@testing-library/jest-dom';

// Mockataan scrollIntoView jotta testit eivät kaadu
window.HTMLElement.prototype.scrollIntoView = function () {};