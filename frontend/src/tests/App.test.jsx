import { render, screen } from "@testing-library/react";
import App from "../App";
import React from "react";

describe("Chat UI", () => {
  test("renders Chat title", () => {
    render(<App />);
    const title = screen.getByText(/Chat/i);
    expect(title).toBeInTheDocument();
  });
});
