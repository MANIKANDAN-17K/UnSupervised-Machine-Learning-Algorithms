# Install mlxtend if not already installed
!pip install mlxtend -q

# Import libraries
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Create realistic Amazon shopping cart data
# Each list = one customer's purchase

transactions = [
    ['Laptop', 'Mouse', 'Laptop Bag'],
    ['Laptop', 'Mouse', 'USB Cable', 'Laptop Bag'],
    ['Phone', 'Phone Case', 'Screen Protector'],
    ['Phone', 'Phone Case', 'Charger'],
    ['Laptop', 'Mouse', 'Keyboard'],
    ['Phone', 'Charger', 'Earphones'],
    ['Laptop', 'Mouse', 'Laptop Bag', 'USB Cable'],
    ['Camera', 'Memory Card', 'Camera Bag'],
    ['Camera', 'Memory Card', 'Tripod'],
    ['Laptop', 'Mouse', 'USB Cable'],
    ['Phone', 'Phone Case', 'Screen Protector', 'Charger'],
    ['Tablet', 'Tablet Case', 'Stylus'],
    ['Laptop', 'Keyboard', 'Mouse'],
    ['Camera', 'Memory Card'],
    ['Phone', 'Charger', 'Phone Case'],
    ['Laptop', 'Mouse', 'Laptop Bag', 'Keyboard'],
    ['Camera', 'Tripod', 'Memory Card', 'Camera Bag'],
    ['Phone', 'Earphones'],
    ['Laptop', 'Mouse'],
    ['Phone', 'Phone Case', 'Screen Protector']
]

print("="*70)
print("AMAZON SHOPPING CART ANALYSIS")
print("="*70)
print(f"\n📊 Total Orders Analyzed: {len(transactions)}")
print("\n🛒 Sample Shopping Carts:")
for i, cart in enumerate(transactions[:5], 1):
    print(f"   Order {i}: {', '.join(cart)}")

# Convert to one-hot encoded format
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)

print("\n" + "="*70)
print("STEP 1: FIND FREQUENT ITEMS (What people buy often)")
print("="*70)

# Apply Apriori - find items bought in at least 20% of orders
frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
frequent_itemsets_sorted = frequent_itemsets.sort_values('support', ascending=False)

print("\n📦 Popular Items (appear in 20%+ of orders):")
for idx, row in frequent_itemsets_sorted.iterrows():
    items = ', '.join(list(row['itemsets']))
    percentage = row['support'] * 100
    print(f"   • {items}: {percentage:.1f}% of orders")

# Generate association rules
print("\n" + "="*70)
print("STEP 2: FIND 'FREQUENTLY BOUGHT TOGETHER' PATTERNS")
print("="*70)

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
rules = rules.sort_values('lift', ascending=False)

print(f"\n🔗 Found {len(rules)} strong buying patterns!\n")

# Display top rules
for idx, row in rules.head(10).iterrows():
    antecedent = ', '.join(list(row['antecedents']))
    consequent = ', '.join(list(row['consequents']))
    
    print(f"Rule {idx + 1}: If customer buys [{antecedent}]")
    print(f"         → They also buy [{consequent}]")
    print(f"   📈 Confidence: {row['confidence']*100:.1f}% ({int(row['confidence']*100)} out of 100 people)")
    print(f"   💡 Lift: {row['lift']:.2f}x more likely than random")
    print()

# Visualize the rules
if len(rules) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Support vs Confidence (colored by Lift)
    scatter = axes[0, 0].scatter(rules['support'], rules['confidence'],
                                 s=rules['lift']*100, c=rules['lift'],
                                 cmap='RdYlGn', alpha=0.6, edgecolors='black')
    axes[0, 0].set_xlabel('Support', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Confidence', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Support vs Confidence (size/color = Lift)', fontsize=13, fontweight='bold')
    axes[0, 0].grid(alpha=0.3)
    plt.colorbar(scatter, ax=axes[0, 0], label='Lift')
    
    # Plot 2: Top rules by Lift
    top_rules = rules.nlargest(8, 'lift')
    rule_labels = [f"{', '.join(list(r['antecedents']))} → {', '.join(list(r['consequents']))}" 
                   for _, r in top_rules.iterrows()]
    y_pos = np.arange(len(top_rules))
    axes[0, 1].barh(y_pos, top_rules['lift'], color='steelblue', edgecolor='black')
    axes[0, 1].set_yticks(y_pos)
    axes[0, 1].set_yticklabels(rule_labels, fontsize=9)
    axes[0, 1].set_xlabel('Lift', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Top 8 Rules by Lift', fontsize=13, fontweight='bold')
    axes[0, 1].axvline(x=1, color='red', linestyle='--', linewidth=2)
    axes[0, 1].grid(alpha=0.3, axis='x')
    
    # Plot 3: Confidence of top rules
    axes[1, 0].barh(y_pos, top_rules['confidence']*100, color='orange', edgecolor='black')
    axes[1, 0].set_yticks(y_pos)
    axes[1, 0].set_yticklabels(rule_labels, fontsize=9)
    axes[1, 0].set_xlabel('Confidence (%)', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Confidence: How Often This Actually Happens', fontsize=13, fontweight='bold')
    axes[1, 0].grid(alpha=0.3, axis='x')
    
    # Plot 4: Item popularity
    item_counts = {}
    for transaction in transactions:
        for item in transaction:
            item_counts[item] = item_counts.get(item, 0) + 1
    
    items = list(item_counts.keys())
    counts = list(item_counts.values())
    sorted_indices = np.argsort(counts)[::-1][:10]
    
    axes[1, 1].bar(range(len(sorted_indices)), [counts[i] for i in sorted_indices],
                  color='green', edgecolor='black', alpha=0.7)
    axes[1, 1].set_xticks(range(len(sorted_indices)))
    axes[1, 1].set_xticklabels([items[i] for i in sorted_indices], rotation=45, ha='right')
    axes[1, 1].set_ylabel('Number of Orders', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Top 10 Most Popular Items', fontsize=13, fontweight='bold')
    axes[1, 1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()

# Amazon's business insights
print("="*70)
print("HOW AMAZON USES THIS DATA:")
print("="*70)

if len(rules) > 0:
    top_rule = rules.iloc[0]
    item1 = ', '.join(list(top_rule['antecedents']))
    item2 = ', '.join(list(top_rule['consequents']))
    
    print(f"\n🎯 Strongest Pattern: [{item1}] → [{item2}]")
    print(f"   Confidence: {top_rule['confidence']*100:.1f}%")
    print(f"   Lift: {top_rule['lift']:.2f}x more likely")
    
    print(f"\n💰 BUSINESS ACTIONS:")
    print(f"   1. Show '{item2}' in 'Frequently Bought Together' section")
    print(f"   2. Bundle deal: Buy {item1} + {item2} = Save 10%")
    print(f"   3. Email: 'You bought {item1}, you might need {item2}'")
    print(f"   4. Add-to-cart suggestion: 'Customers also bought {item2}'")
    print(f"   5. Place {item2} near {item1} in Amazon warehouse for faster shipping")

print("\n" + "="*70)
print("REAL-WORLD IMPACT:")
print("="*70)
print("\n📈 Amazon reports that 35% of revenue comes from recommendations")
print("   powered by algorithms like Apriori!")
print("\n✅ This is why you see:")
print("   • 'Frequently Bought Together' boxes")
print("   • 'Customers who bought this also bought...'")
print("   • Bundle deals and combo offers")
print("   • Personalized email recommendations")
