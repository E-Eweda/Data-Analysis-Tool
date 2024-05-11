import tkinter as tk
from tkinter import *
import tkinter.ttk as ttk
from tkinter import filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import ImageTk,Image
from apyori import  apriori

def split_customers(dataset,n):
    # Load the dataset and preprocess the data
    data = pd.read_csv(dataset)
    customer_names = data["customer"].values
    customer_ages = data["age"].values
    customer_totals = data["total"].values
    customer_data = np.column_stack([customer_totals, customer_ages])

    # Use k-means clustering to split customers into n groups
    kmeans = KMeans(n_clusters=n, n_init=20)
    kmeans.fit(customer_data)

    # Create a new window
    window = tk.Toplevel(root)
    window.config(bg="#001253")
    # Use a custom font and color scheme
    custom_font = ("Arial", 14)
    bg_color = "#fff"
    text_color = "#000"
    button_color = "#00f"

    # Create a function to display the plot
    def display_plot():
        # Extract the cluster labels and the data points
        labels = kmeans.labels_
        points = customer_data
        # Create a scatter plot of the data points, coloring each point according to its cluster label
        plt.scatter(points[:, 0], points[:, 1], c=labels, cmap='viridis')
        plt.xlabel('Total')     
        plt.ylabel('Age')
        plt.title('K-Means Clustering')
        plt.show()

    # Use a themed button to display the plot
    tk.Button(window, text="Show plot", command=display_plot, font=custom_font, bg=button_color, fg="white").pack()

    # Create a scrollbar
    scrollbar = tk.Scrollbar(window)
    scrollbar.pack(side="right", fill="y")

    # Use a themed text box to display the output
    text = tk.Text(window, yscrollcommand=scrollbar.set, width=31, font=custom_font, bg=bg_color, fg=text_color)
    text.pack(side="left", fill="both")
    scrollbar.config(command=text.yview)

    # Insert the output into the text box
    text.insert("end", "Name\tAge\tTotal\tCluster\n")
    for i in range(len(customer_data)):
        name = customer_names[i]
        age = customer_ages[i]
        total = customer_totals[i]
        cluster = kmeans.labels_[i]
        text.insert("end", f"{name}\t{age}\t{total}\t{cluster}\n")
def find_optimal_clusters(dataset):
    # Load the dataset and preprocess the data
    data = pd.read_csv(dataset)
    # Determine the optimal number of clusters using the elbow method
    inertias = []
    for k in range(1, 15):
        km = KMeans(n_clusters=k, n_init=20)
        km = km.fit(data[['age', 'total']])
        inertias.append(km.inertia_)

    plt.plot(range(1, 15), inertias, 'bo-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertias')
    plt.title('Elbow Method For Optimal k')
    plt.show()
def mine_association_rules(dataset):
    # Use a custom font and color scheme
    custom_font = ("Arial", 14)
    bg_color = "#fff"
    text_color = "#000"
    button_color = "#00f"

    # Load and preprocess the data as before
    df1 = pd.read_csv(dataset, header=None)
    df1.columns = ["items", "count", "total", "rnd", "customer", "age", "city", "paymentType"]
    records = []
    itemDataSet = pd.DataFrame(df1['items'])
    itemDataSet = pd.concat([itemDataSet['items'], itemDataSet['items'].str.split(',', expand=True)], axis=1)
    for i in range(0, 9835):
        records.append([str(itemDataSet.values[i, j]) for j in range(0, 33)])
    new_records = []
    temp = []
    for i in range(0, 9835):
        for j in range(0, len(records[i])):
            if records[i][j] != 'None':
                temp.append(records[i][j])
        new_records.append(temp)
        temp = []
    del records
    del temp
    # Create a new window
    window = tk.Toplevel(root)
    window.config(bg="#001253")
    window.geometry("350x200")
    # Create labels for the minimum support and confidence values
    tk.Label(window, text="Minimum Support:",bg="#001253",fg="white",font=custom_font).grid(row=0, column=1, padx=5, pady=5)
    support_entry = tk.Entry(window)
    support_entry.grid(row=0, column=2, padx=5, pady=5)
    tk.Label(window, text="Minimum Confidence:",bg="#001253",fg="white",font=custom_font).grid(row=1, column=1, padx=5, pady=5)
    confidence_entry = tk.Entry(window)
    confidence_entry.grid(row=1, column=2, padx=5, pady=5)
    tk.Label(window, text="Minimum Lift:", bg="#001253", fg="white", font=custom_font).grid(row=2, column=1, padx=5, pady=5)
    min_lift_entry = tk.Entry(window)
    min_lift_entry.grid(row=2, column=2, padx=5, pady=5)
    tk.Label(window, text="Minimum Length:", bg="#001253", fg="white", font=custom_font).grid(row=3, column=1, padx=5, pady=5)
    min_length_entry = tk.Entry(window)
    min_length_entry.grid(row=3, column=2, padx=5, pady=5)


    def mine():
        # Get the minimum support and confidence values from the Entry widgets
        min_support = float(support_entry.get())
        min_confidence = float(confidence_entry.get())
        min_lift=float(min_lift_entry.get())
        min_length=float(min_length_entry.get())
        # Mine association rules using the Apriori algorithm
        frequent_itemsets = apriori(new_records, min_support=min_support, min_confidence=min_confidence, min_lift=min_lift,min_length=min_length)
        rules = list(frequent_itemsets)


        # Insert the output into the text box
        def inspect(output):
            lhs = [tuple(result[2][0][0])[0] for result in output]
            rhs = [tuple(result[2][0][1])[0] for result in output]
            support = [result[1] for result in output]
            confidence = [result[2][0][2] for result in output]
            lift = [result[2][0][3] for result in output]
            return list(zip(lhs, rhs, support, confidence, lift))

        def display_rules(rules):
            # Create a new window
            window = tk.Toplevel(root)
            window.geometry("850x230")
            window.title("Association Rules")
            # Create a Treeview widget to display the rules
            tree = ttk.Treeview(window, columns=("lhs", "rhs", "support", "confidence", "lift"), show="headings")
            tree.pack(side="left")

            # Set the column headings
            tree.column("lhs", width=200, anchor="center")
            tree.column("rhs", width=200, anchor="center")
            tree.column("support", width=150, anchor="center")
            tree.column("confidence", width=150, anchor="center")
            tree.column("lift", width=150, anchor="center")
            tree.heading("lhs", text="Left Hand Side")
            tree.heading("rhs", text="Right Hand Side")
            tree.heading("support", text="Support")
            tree.heading("confidence", text="Confidence")
            tree.heading("lift", text="Lift")

            # Insert the rules into the Treeview widget
            for i, rule in enumerate(rules):
                tree.insert("", "end", values=rule)
        display_rules(inspect(rules))
    # Create a button to mine the association rules
    tk.Button(window, text="Mine association rules", command=mine, font=custom_font, bg=button_color, fg="white").place(x=80,y=160)
# Use a custom font and color scheme
custom_font = ("Arial", 14)
bg_color = "#000000"
text_color = "#000"
button_color = "#00f"
# Create Tkinter GUI
root = tk.Tk()

root.iconbitmap("./assets/logo.ico")
root.title("Data Analysis")
image_0=Image.open("./assets/background.jpg")
bck_end=ImageTk.PhotoImage(image_0)
lbl=Label(root,image=bck_end)
lbl.place(x=-2,y=-2)
    # Create a custom font and color scheme
custom_font = ("Arial", 12)
# Add a label for the dataset file
tk.Label(root, text="Dataset file:",font="white",bg="#001253",fg="white").grid(row=0, column=0, padx=5, pady=5)

# Add a text box for dataset file path
file_input = tk.Entry(root,width=30)
file_input.grid(row=0, column=1, padx=5, pady=5)

# Add a button to browse for a dataset file
tk.Button(root, text="Browse", command=lambda: file_input.insert(0, filedialog.askopenfilename()), font=custom_font, bg=button_color, fg="white").grid(row=0, column=2, padx=5, pady=5)
# Add a label and entry field for the min_threshold value
Split_Clusters_label = tk.Label(text="Number of Cluster:",font=custom_font,fg="white",bg="#001253")
Split_Clusters_label.grid(row=2, column=0, padx=5, pady=5)
Split_Clusters_entry = tk.Entry()
Split_Clusters_entry.grid(row=2, column=1, padx=5, pady=5)

# Add a button to find optimal number of clusters
Optimal_Clusters_button=tk.Button(text="Optimal Clusters", command=lambda:find_optimal_clusters(file_input.get()), font=custom_font, fg="white",bg=button_color)
Optimal_Clusters_button.grid(row=1, column=0, columnspan=3, padx=5, pady=5)
# Add a button to split customers into groups
tk.Button(root, text="Split Customers", command=lambda: split_customers(file_input.get(),int(Split_Clusters_entry.get())), font=custom_font, bg=button_color, fg="white").grid(row=2, column=2, columnspan=3, padx=5, pady=5)

# Add a button to run the 'mine_association_rules' function
association_button = tk.Button(text="Mine Association Rules", command=lambda:mine_association_rules(file_input.get()), font=custom_font, bg=button_color, fg="white")
association_button.grid(row=3, column=0, columnspan=3, padx=5, pady=5)
# Run the tkinter event loop
root.mainloop()