# GPULAB

A Jekyll-based website for GPU computing resources and documentation.

## Prerequisites

- Ruby (version 2.7 or higher)
- Bundler gem
- Jekyll gem

## Installation

Clone or navigate to the project directory:
```bash
cd /path/to/GPULAB
```

Install dependencies:
```bash
bundle install
```

## Development

### Running the Local Server
To start the development server:
```bash
bundle exec jekyll serve
```
The site will be available at: http://localhost:4000

### Additional Server Options

With live reload (auto-refresh on changes):
```bash
bundle exec jekyll serve --livereload
```

Include draft posts:
```bash
bundle exec jekyll serve --drafts
```

Use different port:
```bash
bundle exec jekyll serve --port 3000
```

Watch for changes without serving:
```bash
bundle exec jekyll build --watch
```

## Project Structure
```
GPULAB/
├── _cuda/              # CUDA-related content
├── _includes/          # Reusable HTML snippets
├── _layouts/           # Page templates
├── _site/              # Generated site (auto-generated)
├── assets/             # Static assets
│   ├── images/         # Image files
│   ├── js/             # JavaScript files
│   └── main.scss       # Main stylesheet
├── Gemfile             # Ruby dependencies
├── _config.yml         # Jekyll configuration
└── index.markdown      # Homepage content
```

## Building for Production
To build the site for production:
```bash
bundle exec jekyll build
```
The built site will be in the `_site/` directory.

## Troubleshooting

### Common Issues

Bundle command not found:
```bash
gem install bundler
```

Jekyll command not found:
```bash
gem install jekyll
```

Permission errors on macOS:
```bash
sudo gem install jekyll bundler
```

Port already in use:
```bash
bundle exec jekyll serve --port 4001
```

### Clean Build
If you encounter issues, try cleaning and rebuilding:
```bash
bundle exec jekyll clean
bundle exec jekyll build
```

## Contributing

1. Make your changes
2. Test locally with `bundle exec jekyll serve`
3. Commit and push your changes

## Notes

- The `_site/` directory is auto-generated and should not be edited directly
- Configuration changes in `_config.yml` require a server restart
- Live reload may not work with all browsers; try refreshing manually if neededULAB
A Jekyll-based website for GPU computing resources and documentation.
Prerequisites

Ruby (version 2.7 or higher)
Bundler gem
Jekyll gem

Installation

Clone or navigate to the project directory:
bashcd /path/to/GPULAB

Install dependencies:
bashbundle install


Development
Running the Local Server
To start the development server:
bashbundle exec jekyll serve
The site will be available at: http://localhost:4000
Additional Server Options

With live reload (auto-refresh on changes):
bashbundle exec jekyll serve --livereload

Include draft posts:
bashbundle exec jekyll serve --drafts

Use different port:
bashbundle exec jekyll serve --port 3000

Watch for changes without serving:
bashbundle exec jekyll build --watch


Project Structure
GPULAB/
├── _cuda/              # CUDA-related content
├── _includes/          # Reusable HTML snippets
├── _layouts/           # Page templates
├── _site/              # Generated site (auto-generated)
├── assets/             # Static assets
│   ├── images/         # Image files
│   ├── js/             # JavaScript files
│   └── main.scss       # Main stylesheet
├── Gemfile             # Ruby dependencies
├── _config.yml         # Jekyll configuration
└── index.markdown      # Homepage content
Building for Production
To build the site for production:
bashbundle exec jekyll build
The built site will be in the _site/ directory.
Troubleshooting
Common Issues

Bundle command not found:
bashgem install bundler

Jekyll command not found:
bashgem install jekyll

Permission errors on macOS:
bashsudo gem install jekyll bundler

Port already in use:
bashbundle exec jekyll serve --port 4001


Clean Build
If you encounter issues, try cleaning and rebuilding:
bashbundle exec jekyll clean
bundle exec jekyll build
Contributing

Make your changes
Test locally with bundle exec jekyll serve
Commit and push your changes

Notes

The _site/ directory is auto-generated and should not be edited directly
Configuration changes in _config.yml require a server restart
Live reload may not work with all browsers; try refreshing manually if needed