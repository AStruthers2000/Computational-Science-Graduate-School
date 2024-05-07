#include "BTree.h"

Node* traverse_tree(Node* head, const int key)
{
    auto current = head;
    while(!current->_is_leaf)
    {
        if(key < current->get_first_key())
        {
            current = current->get_left_child();
        }
        else if (key > current->get_last_key())
        {
            current = current->get_right_child();
        }
        else
        {
            current = current->get_middle_child();
        }
    }
    return current;
}

void handle_key_overflow(Node* current)
{
    while(current->_num_keys >= MAX_KEYS_ALLOWED)
    {
        const int middle_key = current->get_middle_key();
        current->remove_middle_key();
        if(current->parent == nullptr)
        {
            current->parent = new Node();
            current->parent->insert_child(current);
        }
        current->parent->insert_key(middle_key);
        
        Node *left, *right;
        break_node(current, left, right);
        current->parent->remove_child(current);
        current->parent->insert_child(left);
        current->parent->insert_child(right);
        current = current->parent;
    }
}

void break_node(const Node* current, Node* &left_node, Node* &right_node)
{
    left_node = new Node();
    right_node = new Node();
    left_node->parent = current->parent;
    right_node->parent = current->parent;
    left_node->insert_key(current->get_first_key());
    right_node->insert_key(current->get_last_key());
    if(!current->_is_leaf)
    {
        assert(current->children.size() == 4);
        left_node->insert_child(current->children[0]);
        left_node->insert_child(current->children[1]);
        right_node->insert_child(current->children[2]);
        right_node->insert_child(current->children[3]);
    }
}

Node* find_head(Node* current)
{
    auto temp = current;
    while(temp->parent!=nullptr)
    {
        temp = temp->parent;
    }
    return temp;
}

void insert_key(Node* &head, const int k)
{
    const auto current = traverse_tree(head, k);
    current->insert_key(k);
    handle_key_overflow(current);
    head = find_head(current);
}

int main(int argc, char* argv[])
{
    //const vector<int> keys = {10, 9, 4, 11, 26, 9001, 100000000};
    vector<int> keys = {15,10,20,5,13,17,30,3,8,9,11,14,16,18,25};

    shuffle(keys.begin(), keys.end(), mt19937(random_device()()));

    auto head = new Node();

    for(const auto& k : keys)
    {
        insert_key(head, k);
    }

    insert_key(head, 35);
    
    return 0;
}
