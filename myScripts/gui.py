import tkinter as tk
import asyncio

class BotGUI:
    def __init__(self, master_bot, loop):
        self.master_bot = master_bot
        self.loop = loop
        self.root = tk.Tk()
        self.root.title("StockBot Manager")
        self.buttons = {}  # Store button references
        self.create_widgets()

    def create_widgets(self):
        for i, symbol in enumerate(self.master_bot.symbols):
            # Left-aligned label
            tk.Label(
                self.root,
                text=symbol,
                font=('Arial', 12),
                anchor='w',
                justify='left',
                width=25
            ).grid(row=i, column=0, padx=10, pady=5, sticky='w')

            # Terminate button
            btn = tk.Button(
                self.root,
                text="Terminate",
                font=('Arial', 10),
                fg='white',
                bg='red',
                command=lambda sym=symbol: self.disable_and_terminate(sym)
            )
            btn.grid(row=i, column=1, padx=10, pady=5)
            self.buttons[symbol] = btn

    def run(self):
        self.root.mainloop()

    def disable_and_terminate(self, symbol):
        btn = self.buttons.get(symbol)
        if btn:
            btn.config(state='disabled', text="Terminated", bg='gray')
        asyncio.run_coroutine_threadsafe(
            self.terminate_bot(symbol),
            self.loop
        )

    async def terminate_bot(self, symbol):
        bot = self.master_bot.bots.get(symbol)
        if bot:
            await bot.terminate()
            print(f"Terminated bot for {symbol}")
        else:
            print(f"No bot found for {symbol}")
