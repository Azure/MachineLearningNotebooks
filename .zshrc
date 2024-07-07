#
# ~/.zshrc
#

# If not running interactively, don't do anything
[[ $- != *i* ]] && return


alias ls='ls --color=auto'
PS1='[\u@\h \W]\$ '


#aliases
alias pac="sudo pacman -S"
alias pacs="sudo pacman -Ss"
alias pacu="sudo pacman -Syy"
alias update="sudo pacman -Syu"

