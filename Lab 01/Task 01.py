class GroceryManager:
    def __init__(self):
        self.items = {}

    def add_item(self, item, quantity, price):
        self.items[item] = {"quantity": quantity, "price": price}

    def remove_item(self, item):
        if item not in self.items:
            print(f"Error: '{item}' does not exist in the list.")
            return
        del self.items[item]

    def view_list(self):
        for item, details in self.items.items():
            print(f"{item}: Qty={details['quantity']}, Price={details['price']}")

    def calculate_total(self):
        return sum(d["quantity"] * d["price"] for d in self.items.values())


gm = GroceryManager()
gm.add_item("Apple", 4, 50)
gm.add_item("Milk", 2, 180)
gm.remove_item("Bread")
gm.view_list()
print("Total:", gm.calculate_total())
