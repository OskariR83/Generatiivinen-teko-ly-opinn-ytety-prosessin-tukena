import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import App from "../App";

global.fetch = jest.fn(() =>
  Promise.resolve({
    ok: true,
    json: () => Promise.resolve({ reply: "Mocked response" }),
  })
);

describe("Chat message sending", () => {
  test("typing and sending a message renders it in chat", async () => {
    render(<App />);

    // Etsitään tekstikenttä placeholderin perusteella
    const input = screen.getByPlaceholderText(/Kirjoita viesti/i);
    // Etsitään "Lähetä viesti" -nappi
    const sendBtn = screen.getByText(/Lähetä viesti/i);

    // Simuloidaan viestin kirjoittaminen tekstikenttään
    fireEvent.change(input, { target: { value: "Moikka" } });
    // Simuloidaan Lähetä-napin painallus
    fireEvent.click(sendBtn);

    // Odotetaan että käyttäjän lähettämä viesti ilmestyy DOMiin
    const msg = await screen.findByText("Moikka");
    // Varmistetaan että viesti näkyy chatissa
    expect(msg).toBeInTheDocument();
  });
});
