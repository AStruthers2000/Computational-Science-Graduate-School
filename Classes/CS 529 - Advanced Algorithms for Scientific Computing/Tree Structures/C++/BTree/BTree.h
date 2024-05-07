#pragma once
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
#include <random>

using namespace std;
enum
{
    MAX_KEYS_ALLOWED = 3
};

struct Node;
bool CompareNodeByFirstElement(const Node* me, const Node* other);

struct Node
{
    Node* parent;
    vector<int> keys;
    vector<Node*> children;

    int _num_keys = 0;
    bool _is_leaf = true;

    void insert_key(int n)
    {
        keys.push_back(n);
        _num_keys++;
        sort(keys.begin(), keys.end());
    }

    void remove_middle_key()
    {
        keys = {*keys.begin(), *(keys.end()-1)};
        _num_keys--;
    }

    void insert_child(Node* child)
    {
        _is_leaf = false;
        child->parent = this;
        children.push_back(child);
        sort(children.begin(), children.end(), CompareNodeByFirstElement);
    }

    void remove_child(const Node* child)
    {
        int i = 0;
        for(const auto& c : children)
        {
            if(child == c)
            {
                break;
            }
            i++;
        }
        children.erase(children.begin()+i);
        if(children.empty()) _is_leaf = true;
    }

    int get_first_key() const
    {
        return *keys.begin();
    }

    int get_last_key() const
    {
        return *(keys.end()-1);
    }

    int get_middle_key() const
    {
        assert(keys.size() == MAX_KEYS_ALLOWED);
        return *(keys.begin()+1);
    }

    Node* get_left_child() const
    {
        return *children.begin();
    }

    Node* get_right_child() const
    {
        return *(children.end()-1);
    }

    Node* get_middle_child() const
    {
        assert(children.size() == MAX_KEYS_ALLOWED);
        return *(children.begin()+1);
    }
};

inline bool CompareNodeByFirstElement(const Node* me, const Node* other)
{
    return me->get_first_key() < other->get_first_key();
}


Node* traverse_tree(Node* head, const int key);
void handle_key_overflow(Node* current);
void break_node(const Node* current, Node* &left_node, Node* &right_node);
Node* find_head(Node* current);
void insert_key(Node* &head, const int k);
