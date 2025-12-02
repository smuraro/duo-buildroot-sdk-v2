#!/bin/bash

#TOP_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
echo "---> $TOP_DIR"
BUILDROOT_DIR=$(basename $(realpath $TOP_DIR/buildroot*))
echo "-->> $BUILDROOT_DIR"
function rsync_dir()
{
    if [ ! -d $1 ]; then
        echo "$1 not exist"
        return
    fi
    mkdir -p $PROJECT_OUT/$2
    echo "rsync $1 -> $PROJECT_OUT/$2"; rsync -a --exclude='.git' $1 $PROJECT_OUT/$2 || exit 1
}

###################################
# patch externals
###################################
echo ">>>> $EXTERNAL | $BUILDROOT_DIR <<<<"
rsync_dir $EXTERNAL/buildroot/ $TOP_DIR/$BUILDROOT_DIR/
