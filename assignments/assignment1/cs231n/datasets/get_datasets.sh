# Get CIFAR10
wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzvf cifar-10-python.tar.gz
rm cifar-10-python.tar.gz 

# <Program Outline>
# 1. Download the data from University of Toronto's server.
# 2. Use the `tar` command to download the data.
# 3. Delete the zip file because it's unnecessary.
#
# <References>
# wget: A GNU project computer program that retrieves content from web servers. In this case, the
#       CIFAR data is contained on UofT's server.
#       https://linux.die.net/man/1/wget
# tar: Saves many files together in a single tape or disk archive, and can also restore those individual
#      files. Basically a "zipping" command. Short for "tape archive."
#        [-x]: Extract files from an archive.
#        [-z]: filter the archive through gzip.
#        [-v]: verbosely list files processed.
#        [-f]: use archive file or device ARCHIVE. Tells `tar` the next parameter is the file or archive name.
#      https://linux.die.net/man/1/tar
# rm: Removes specified files. By default, it doesn't remove directories, but you could specify the [-r]
#     option to tell the program to remove directories by recursively deleting their individual contents.
#     https://linux.die.net/man/1/rm
