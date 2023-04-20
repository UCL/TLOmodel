.PHONY: all clean

platforms = linux-64 osx-64 osx-arm64 win-64
base_in_files = base-in.yml
dev_in_files = $(base_in_files) docs-in.yml dev-in.yml
docs_in_files = docs-in.yml
base_lock_prefix = conda-base
dev_lock_prefix = conda-dev
docs_lock_prefix = conda-docs
base_lock_files = $(foreach p, $(platforms), $(base_lock_prefix)-$(p).lock)
dev_lock_files = $(foreach p, $(platforms), $(dev_lock_prefix)-$(p).lock)
docs_lock_files = $(foreach p, $(platforms), $(docs_lock_prefix)-$(p).lock)

define run_conda_lock =
conda-lock --kind explicit $(foreach f, $^, -f $(f)) $(foreach p, $(platforms), -p $(p))
endef

$(base_lock_files) &: $(base_in_files)
	$(run_conda_lock) --filename-template $(base_lock_prefix)-{platform}.lock

$(dev_lock_files) &: $(dev_in_files)
	$(run_conda_lock) --filename-template $(dev_lock_prefix)-{platform}.lock
	
$(docs_lock_files) &: $(docs_in_files)
	$(run_conda_lock) --filename-template $(docs_lock_prefix)-{platform}.lock

all: $(base_lock_files) $(dev_lock_files) $(docs_lock_files)

clean:
	- rm *.lock
