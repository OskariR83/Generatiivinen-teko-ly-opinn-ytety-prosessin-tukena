import React from "react";
import App from "../App";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";

beforeEach(() => {
  global.fetch = jest.fn(() =>
    Promise.resolve({
      ok: true,
      json: () => Promise.resolve({ answer: "" }),
    })
  );
});

describe("Delete conversation flow", () => {

  test("opens delete modal and confirms deletion", async () => {
    render(<App />);

    // Lisää viesti
    const input = screen.getByPlaceholderText(/Kirjoita viesti/i);
    const sendButton = screen.getByRole("button", { name: /Lähetä viesti/i });

    fireEvent.change(input, { target: { value: "Testiviesti" } });
    fireEvent.click(sendButton);

    // ⬅️ Odotetaan että viesti tulee ruutuun (tämä korjaa act-varoituksen)
    const addedMsg = await screen.findByText("Testiviesti");
    expect(addedMsg).toBeInTheDocument();

    // Klikkaa "Poista keskustelu"
    const deleteButton = screen.getByRole("button", { name: /Poista keskustelu/i });
    fireEvent.click(deleteButton);

    // Modaali näkyy
    expect(await screen.findByText(/Poistetaanko keskustelu/i)).toBeInTheDocument();

    // Klikataan modaalin Poista
    const modalDeleteConfirm = screen.getByRole("button", { name: /^Poista$/i });
    fireEvent.click(modalDeleteConfirm);

    // Odotetaan että viesti häviää DOM:ista
    await waitFor(() => {
      expect(screen.queryByText("Testiviesti")).not.toBeInTheDocument();
    });
  });
});
